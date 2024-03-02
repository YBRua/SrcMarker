import os
import re
import json
import torch
import pickle
import random

from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from models import (
    TransformerEncoderExtractor,
    GRUEncoder,
    MLP2,
)

from logger_setup import setup_evaluation_logger
from data_processing import DataInstance
from data_processing import JsonlWMDatasetProcessor, DynamicWMCollator
from eval_utils import compute_msg_acc
from metrics import calc_code_bleu

from typing import List, Set


def fix_join_artifacts(text: str):
    # remove spaces between dots
    text = re.sub(r"\s?\.\s?(?=\w+)", ".", text)
    # remove spaces between underscores
    text = re.sub(r"_\s?(?=\w+)", "_", text)
    # replace 0X with 0x
    text = re.sub(r"0X", "0x", text)
    return text


def join_lines_default(tokens: List[str], keywords: Set[str]):
    line = ""
    prev = ""
    is_member_access = False
    for tok in tokens:
        if tok == "<unk>":
            if prev[0].isupper():
                tok = "Unk"
            else:
                tok = "Unk"

        if tok == ".":
            is_member_access = True
        elif not tok.isidentifier():
            is_member_access = False

        if (
            prev.isidentifier()
            and tok.isidentifier()
            and prev not in keywords
            and tok not in keywords
        ):
            if tok[0] == "_" or tok[0].isupper():
                line += tok
            else:
                line += f" {tok}"
        elif prev == "." and tok == "*":
            line += tok
        elif is_member_access:
            line += tok
        elif prev == "<" or tok in {"<", ">"}:
            line += tok
        else:
            line += f" {tok}"
        prev = tok

    return line


def join_lines_java_heuristic(tokens: List[str], keywords: Set[str]):
    line = ""
    prev = ""
    is_member_access = False
    for tok in tokens:
        if tok == "<unk>":
            if len(prev) and prev[0].isupper():
                tok = "Unk"
            else:
                tok = "Unk"

        if tok == ".":
            is_member_access = True
        elif not tok.isidentifier():
            is_member_access = False

        if (
            prev.isidentifier()
            and tok.isidentifier()
            and prev not in keywords
            and tok not in keywords
        ):
            if tok[0] == "_" or tok[0].isupper():
                line += tok
            else:
                line += f" {tok}"
        elif prev == "." and tok == "*":
            line += tok
        elif is_member_access:
            line += tok
        elif prev == "<" or tok in {"<", ">"}:
            line += tok
        elif prev == "@":
            line += tok
        elif prev in keywords:
            line += f" {tok}"
        else:
            line += f" {tok}"
        prev = tok

    return line


def join_lines(tokens: List[str], keywords: Set[str], lang: str):
    if lang == "java" or lang == "javascript":
        return join_lines_java_heuristic(tokens, keywords)
    else:
        return join_lines_default(tokens, keywords)


def parse_args_for_adaptive_removal_attack():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        choices=[
            "github_c_funcs",
            "github_java_funcs",
            "csn_java",
            "csn_js",
            "mbjp",
            "mbjsp",
        ],
        default="github_java_funcs",
    )
    parser.add_argument("--lang", choices=["cpp", "java", "javascript"], default="java")
    parser.add_argument(
        "--dataset_dir",  # path to original dataset
        type=str,
        default="./datasets/github_java_funcs",
    )
    parser.add_argument(
        "--original_test_outputs",  # path to the original output on testset (jsonl)
        type=str,
        default="./datasets/dewatermark/test.jsonl",
    )
    parser.add_argument(
        "--attack_results",  # path to dewatermarking attack results (json)
        type=str,
        default="./datasets/dewatermark/attack.json",
    )
    parser.add_argument("--n_bits", type=int, default=4)

    parser.add_argument("--log_prefix", type=str, default="")
    parser.add_argument("--model_arch", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--checkpoint_path", type=str, default="./ckpts/my_model")
    parser.add_argument("--output_txt", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args_for_adaptive_removal_attack()
    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    ATTACK_RESULTS = args.attack_results
    ORIGINAL_TEST_OUTPUTS = args.original_test_outputs
    SEED = args.seed
    CKPT_PATH = args.checkpoint_path
    MODEL_ARCH = args.model_arch
    N_BITS = args.n_bits

    KEYWORDS = set(open(f"./metrics/keywords/{LANG}.txt", "r").read().split("\n"))

    logger = setup_evaluation_logger(args, prefix="dewatermark")
    logger.info(args)
    DEVICE = torch.device("cuda")

    # seed everything
    torch.manual_seed(SEED)
    random.seed(SEED)

    # load original test outputs (for ground truth watermarks)
    with open(ORIGINAL_TEST_OUTPUTS, "r") as fi:
        test_output_objs = [json.loads(line) for line in fi.readlines()]

    # load datasets
    dataset_processor = JsonlWMDatasetProcessor(lang=LANG)
    checkpoint_dict = torch.load(CKPT_PATH, map_location="cpu")
    vocab = checkpoint_dict["vocab"]

    logger.info("Processing original dataset")
    test_instances: List[DataInstance] = dataset_processor.load_jsonl(
        DATASET_DIR, split="test"
    )
    raw_test_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split="test")

    # load attacked tokens and replace original tokens
    with open(ATTACK_RESULTS, "r") as fi:
        attacked_objs = json.load(fi)
    assert len(test_instances) == len(attacked_objs)
    assert len(test_instances) == len(raw_test_objs)
    test_tokens_backup = []
    for instance, attacked in zip(test_instances, attacked_objs):
        test_tokens_backup.append(list(instance.tokens))
        instance.tokens = attacked["tokens"]

    test_dataset = dataset_processor.build_dataset(test_instances, vocab)
    logger.info(f"Vocab size: {len(vocab)}")
    logger.info(f"Test size: {len(test_dataset)}")

    print("building models")
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=DynamicWMCollator(N_BITS)
    )

    logger.info(f"Using {MODEL_ARCH}")
    if MODEL_ARCH == "gru":
        FEATURE_DIM = 768
        encoder = GRUEncoder(
            vocab_size=len(vocab), hidden_size=FEATURE_DIM, embedding_size=FEATURE_DIM
        )
    elif MODEL_ARCH == "transformer":
        FEATURE_DIM = 768
        encoder = TransformerEncoderExtractor(
            vocab_size=len(vocab), embedding_size=FEATURE_DIM, hidden_size=FEATURE_DIM
        )
    wm_decoder = MLP2(output_dim=N_BITS, bn=False, input_dim=FEATURE_DIM)

    print(f"loading checkpoint from {CKPT_PATH}")
    ckpt_save = torch.load(CKPT_PATH, map_location="cpu")
    encoder.load_state_dict(ckpt_save["model"])
    wm_decoder.load_state_dict(ckpt_save["wm_decoder"])

    encoder.to(DEVICE)
    wm_decoder.to(DEVICE)

    encoder.eval()
    wm_decoder.eval()

    n_samples = 0
    tot_acc = 0.0
    tot_msg_acc = 0.0
    codebleus = defaultdict(int)

    repo_wise = DATASET in {"csn_java", "csn_js"}  # only for csn datasets
    repowise_long_msg = defaultdict(list)
    repowise_long_msg_dewm = defaultdict(list)

    print("beginning evaluation")

    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    if args.output_txt:
        fo = open(f"./results/{ckpt_name}_dewm_{DATASET}.txt", "w")

    with torch.no_grad():
        prog = tqdm(test_loader)
        for bid, batch in enumerate(prog):
            (x, lengths, src_mask, instance_ids, wms, wmids) = batch
            B = x.shape[0]
            assert B == 1
            n_samples += 1

            test_output_obj = test_output_objs[bid]
            raw_obj = raw_test_objs[bid]
            instance = test_instances[bid]
            watermark = torch.tensor(
                test_output_obj["watermark"], dtype=torch.long, device=DEVICE
            )
            watermark = watermark.unsqueeze(0)  # B, N_BITS

            x = x.to(DEVICE)
            src_mask = src_mask.to(DEVICE)

            features = encoder(x, lengths, src_mask)
            outputs = wm_decoder(features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            tot_acc += torch.sum(torch.mean((preds == watermark).float(), dim=1)).item()
            tot_msg_acc += compute_msg_acc(preds, watermark, n_bits=N_BITS)

            original_str = " ".join(test_output_obj["original_tokens"])
            test_out_str = " ".join(test_output_obj["after_watermark_tokens"])
            attacked_str = " ".join(instance.tokens)

            logger.info(f"Sample No.{n_samples}")
            logger.info(f"Original     : {original_str}")
            logger.info(f"Watermarked  : {test_out_str}")
            logger.info(f"Dewatermarked: {attacked_str}")
            logger.info(f"Watermark    : {watermark.tolist()}")
            logger.info(f"Prediction   : {preds[0].tolist()}")
            logger.info("=" * 50)

            codebleu_res = calc_code_bleu.evaluate_per_example(
                test_out_str, attacked_str, lang=LANG
            )

            if args.output_txt:
                fixed_str = join_lines(instance.tokens, KEYWORDS, lang=LANG)
                fixed_str = fix_join_artifacts(fixed_str)
                fo.write(f"{fixed_str}\n")

            if repo_wise:
                repo = raw_obj["repo"]
                repowise_long_msg[repo].append(wms[0].tolist())
                repowise_long_msg_dewm[repo].append(preds[0].tolist())

            for key in codebleu_res.keys():
                codebleus[key] += codebleu_res[key]

    logger.info(f"Accuracy: {tot_acc / n_samples:.4f}")
    logger.info(f"Message accuracy: {tot_msg_acc / n_samples:.4f}")
    for key in codebleus.keys():
        logger.info(f"CodeBLEU-{key}: {codebleus[key] / n_samples:.4f}")

    if args.output_txt:
        fo.close()
    if repo_wise:
        pickle.dump(
            [repowise_long_msg, repowise_long_msg_dewm],
            open(f"./results/{ckpt_name}_dewm_long.pkl", "wb"),
        )


if __name__ == "__main__":
    main()
