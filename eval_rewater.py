import os
import json
import torch
import random
import pickle
import tree_sitter
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from models import (
    ConcatApproximator,
    TransformSelector,
    TransformerEncoderExtractor,
    GRUEncoder,
    ExtractGRUEncoder,
    WMLinearEncoder,
    MLP2,
)

from metrics import calc_code_bleu
from code_tokenizer import tokens_to_strings
from eval_utils import compute_msg_acc
from code_transform_provider import CodeTransformProvider
from runtime_data_manager import InMemoryJitRuntimeDataManager
from data_processing import JsonlWMDatasetProcessor, DynamicWMCollator
from logger_setup import setup_evaluation_logger
import mutable_tree.transformers as ast_transformers


def parse_args_for_evaluation():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "github_c_funcs",
            "github_java_funcs",
            "csn_java",
            "csn_js",
            "mbjp",
            "mbjsp",
            "mbcpp",
        ],
        default="github_c_funcs",
    )
    parser.add_argument("--lang", choices=["cpp", "java", "javascript"], default="c")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/github_c_funcs")
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--checkpoint_path", type=str, default="./ckpts/something.pt")
    parser.add_argument("--adv_path", type=str, default="./ckpts/badbad.pt")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_arch", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--shared_encoder", action="store_true")

    parser.add_argument(
        "--var_transform_mode", choices=["replace", "append"], default="replace"
    )

    parser.add_argument("--write_output", action="store_true")

    return parser.parse_args()


def main(args):
    logger = setup_evaluation_logger(args, prefix="rewater")
    logger.info(args)

    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    N_BITS = args.n_bits
    CKPT_PATH = args.checkpoint_path
    ADV_PATH = args.adv_path
    DEVICE = torch.device("cuda")
    MODEL_ARCH = args.model_arch
    SHARED_ENCODER = args.shared_encoder
    VAR_TRANSFORM_MODE = args.var_transform_mode

    # seed
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)

    PARSER_LANG = tree_sitter.Language("parser/languages.so", args.lang)
    ts_parser = tree_sitter.Parser()
    ts_parser.set_language(PARSER_LANG)
    code_transformers = [
        ast_transformers.IfBlockSwapTransformer(),
        ast_transformers.CompoundIfTransformer(),
        ast_transformers.ConditionTransformer(),
        ast_transformers.LoopTransformer(),
        ast_transformers.InfiniteLoopTransformer(),
        ast_transformers.UpdateTransformer(),
        ast_transformers.SameTypeDeclarationTransformer(),
        ast_transformers.VarDeclLocationTransformer(),
        ast_transformers.VarInitTransformer(),
        ast_transformers.VarNameStyleTransformer(),
    ]

    transform_computer = CodeTransformProvider(LANG, ts_parser, code_transformers)

    dataset_processor = JsonlWMDatasetProcessor(LANG)

    checkpoint_dict = torch.load(CKPT_PATH, map_location="cpu")
    vocab = checkpoint_dict["vocab"]
    test_instances = dataset_processor.load_jsonl(DATASET_DIR, split="test")
    raw_test_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split="test")
    test_dataset = dataset_processor.build_dataset(test_instances, vocab)
    print(f"Vocab size: {len(vocab)}")
    print(f"Test size: {len(test_dataset)}")

    if VAR_TRANSFORM_MODE == "replace":
        vmask = vocab.get_valid_identifier_mask()
    else:
        vmask = vocab.get_valid_highfreq_mask(2**N_BITS * 32)

    transform_manager = InMemoryJitRuntimeDataManager(
        transform_computer, test_instances, LANG
    )
    transform_manager.register_vocab(vocab)
    transform_manager.load_transform_mask(
        f"./datasets/feasible_transform_{DATASET}.json"
    )
    transform_manager.load_varname_dict(f"./datasets/variable_names_{DATASET}.json")
    transform_capacity = transform_manager.get_transform_capacity()
    print(f"Transform capacity: {transform_capacity}")

    # build models
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
        rw_encoder = GRUEncoder(
            vocab_size=len(vocab), hidden_size=FEATURE_DIM, embedding_size=FEATURE_DIM
        )
        if SHARED_ENCODER:
            extract_encoder = None
        else:
            extract_encoder = ExtractGRUEncoder(
                vocab_size=len(vocab),
                hidden_size=FEATURE_DIM,
                embedding_size=FEATURE_DIM,
            )
    elif MODEL_ARCH == "transformer":
        FEATURE_DIM = 768
        encoder = TransformerEncoderExtractor(
            vocab_size=len(vocab), embedding_size=FEATURE_DIM, hidden_size=FEATURE_DIM
        )
        rw_encoder = TransformerEncoderExtractor(
            vocab_size=len(vocab), embedding_size=FEATURE_DIM, hidden_size=FEATURE_DIM
        )
        if SHARED_ENCODER:
            extract_encoder = None
        else:
            extract_encoder = TransformerEncoderExtractor(
                vocab_size=len(vocab),
                embedding_size=FEATURE_DIM,
                hidden_size=FEATURE_DIM,
            )

    selector = TransformSelector(
        vocab_size=len(vocab),
        transform_capacity=transform_capacity,
        input_dim=FEATURE_DIM,
        vocab_mask=vmask,
    )
    rw_selector = TransformSelector(
        vocab_size=len(vocab),
        transform_capacity=transform_capacity,
        input_dim=FEATURE_DIM,
        vocab_mask=vmask,
    )
    approximator = ConcatApproximator(
        vocab_size=len(vocab),
        transform_capacity=transform_capacity,
        input_dim=FEATURE_DIM,
        output_dim=FEATURE_DIM,
    )
    wm_encoder = WMLinearEncoder(N_BITS, embedding_dim=FEATURE_DIM)
    rw_wm_encoder = WMLinearEncoder(N_BITS, embedding_dim=FEATURE_DIM)
    wm_decoder = MLP2(output_dim=N_BITS, bn=False, input_dim=FEATURE_DIM)

    print(f"loading checkpoint from {CKPT_PATH}")
    ckpt_save = torch.load(CKPT_PATH, map_location="cpu")
    encoder.load_state_dict(ckpt_save["model"])
    if extract_encoder is not None:
        extract_encoder.load_state_dict(ckpt_save["extract_encoder"])
    wm_encoder.load_state_dict(ckpt_save["wm_encoder"])
    wm_decoder.load_state_dict(ckpt_save["wm_decoder"])
    selector.load_state_dict(ckpt_save["selector"])
    approximator.load_state_dict(ckpt_save["approximator"])

    print(f"loading adversary from {ADV_PATH}")
    ckpt_save = torch.load(ADV_PATH, map_location="cpu")
    rw_encoder.load_state_dict(ckpt_save["model"])
    rw_wm_encoder.load_state_dict(ckpt_save["wm_encoder"])
    rw_selector.load_state_dict(ckpt_save["selector"])

    encoder.to(DEVICE)
    if extract_encoder is not None:
        extract_encoder.to(DEVICE)
    selector.to(DEVICE)
    approximator.to(DEVICE)
    wm_encoder.to(DEVICE)
    wm_decoder.to(DEVICE)
    rw_encoder.to(DEVICE)
    rw_wm_encoder.to(DEVICE)
    rw_selector.to(DEVICE)

    encoder.eval()
    if extract_encoder is not None:
        extract_encoder.eval()
    selector.eval()
    approximator.eval()
    wm_encoder.eval()
    wm_decoder.eval()
    rw_encoder.eval()
    rw_wm_encoder.eval()
    rw_selector.eval()

    n_samples = 0
    tot_acc = 0
    tot_adv_acc = 0
    tot_new_acc = 0

    tot_msg_acc = 0
    tot_adv_msg_acc = 0
    tot_new_msg_acc = 0

    adv_codebleu_res = defaultdict(int)

    repo_wise = DATASET in {"csn_java", "csn_js"}  # only for csn datasets
    repowise_long_msg = defaultdict(list)
    repowise_long_msg_adv = defaultdict(list)

    print("beginning evaluation")
    # eval starts from here
    rewatermarked_objs = []
    with torch.no_grad():
        prog = tqdm(test_loader)
        for bid, batch in enumerate(prog):
            test_obj = raw_test_objs[bid]
            repo = test_obj["repo"] if "repo" in test_obj else None

            (x, lengths, src_mask, instance_ids, wms, wmids) = batch
            wms = wms.float()
            B = x.shape[0]

            x = x.to(DEVICE)
            wms = wms.to(DEVICE)
            wmids = wmids.to(DEVICE)
            src_mask = src_mask.to(DEVICE)

            rw_wms = []
            rw_wmids = []
            # generate another batch of wms
            for wm in wms:
                new_wm = torch.randint(0, 2, (N_BITS,))
                wm_cpu = wm.clone().cpu()
                while torch.all(new_wm == wm_cpu):
                    new_wm = torch.randint(0, 2, (N_BITS,))
                rw_wms.append(new_wm)
                rw_wmids.append(sum([2**i * new_wm[i] for i in range(N_BITS)]))
            rw_wms = torch.stack(rw_wms, dim=0).to(DEVICE)
            rw_wmids = torch.tensor(rw_wmids).to(DEVICE)

            # get style masks
            s_feasible = transform_manager.get_feasible_transform_ids(instance_ids)
            s_feasible_01 = []
            for s_f in s_feasible:
                val = torch.ones(transform_capacity, device=DEVICE).bool()
                val[s_f] = False
                s_feasible_01.append(val)
            s_masks = torch.stack(s_feasible_01, dim=0)

            # simulated watermark encoding
            # get transform and variable substitution predictions
            code_feature = encoder(x, lengths, src_mask)
            wm_feature = wm_encoder(wms)

            vs_output = selector.var_selector_forward(code_feature, wm_feature)
            vs_ids = torch.argmax(vs_output, dim=1).tolist()

            ss_output = selector.transform_selector_forward(
                code_feature, wm_feature, transform_mask=s_masks
            )
            ss_ids = torch.argmax(ss_output, dim=1).tolist()

            # transform code
            ori_instances = transform_manager.get_original_instances(instance_ids)
            t_instances, updates = transform_manager.varname_transform_on_instances(
                ori_instances, vs_ids, mode=VAR_TRANSFORM_MODE
            )
            t_instances = transform_manager.transform_on_instances(t_instances, ss_ids)

            # simulated decoding process
            dec_x, dec_l, dec_m = transform_manager.load_to_tensor(t_instances)
            dec_x = dec_x.to(DEVICE)
            dec_m = dec_m.to(DEVICE)

            if extract_encoder is not None:
                t_features = extract_encoder(dec_x, dec_l, dec_m)
            else:
                t_features = encoder(dec_x, dec_l, dec_m)
            outputs = wm_decoder(t_features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            tot_acc += torch.sum(torch.mean((preds == wms).float(), dim=1)).item()
            tot_msg_acc += compute_msg_acc(preds, wms, n_bits=args.n_bits)

            # re-watermarking attack
            rw_code_feature = rw_encoder(dec_x, dec_l, dec_m)
            rw_wm_feature = rw_wm_encoder(rw_wms.float())

            rw_vs_output = rw_selector.var_selector_forward(
                rw_code_feature, rw_wm_feature
            )
            rw_vs_ids = torch.argmax(rw_vs_output, dim=1)

            rw_ss_output = rw_selector.transform_selector_forward(
                rw_code_feature, rw_wm_feature, transform_mask=s_masks
            )
            rw_ss_ids = torch.argmax(rw_ss_output, dim=1)

            # transform code, again
            rw_instances = transform_manager.transform_on_instances(
                t_instances, rw_ss_ids.tolist()
            )
            rw_instances, rw_updates = transform_manager.rewatermark_varname_transform(
                rw_instances, rw_vs_ids.tolist(), updates, mode=VAR_TRANSFORM_MODE
            )

            # simulated decoding on re-watermarked code
            rw_x, rw_l, rw_m = transform_manager.load_to_tensor(rw_instances)
            rw_x = rw_x.to(DEVICE)
            rw_m = rw_m.to(DEVICE)
            if extract_encoder is not None:
                rw_features = extract_encoder(rw_x, rw_l, rw_m)
            else:
                rw_features = encoder(rw_x, rw_l, rw_m)
            rw_outputs = wm_decoder(rw_features)
            rw_probs = torch.sigmoid(rw_outputs)
            rw_preds = (rw_probs > 0.5).long()

            tot_adv_acc += torch.sum(
                torch.mean((rw_preds == wms).float(), dim=1)
            ).item()
            tot_adv_msg_acc += compute_msg_acc(rw_preds, wms, n_bits=args.n_bits)

            tot_new_acc += torch.sum(
                torch.mean((rw_preds == rw_wms).float(), dim=1)
            ).item()
            tot_new_msg_acc += compute_msg_acc(rw_preds, rw_wms, n_bits=args.n_bits)

            # log results
            ori_instances = transform_manager.get_original_instances(instance_ids)
            for i in range(B):
                if repo_wise:
                    repowise_long_msg[repo].extend(wms[i].tolist())
                    repowise_long_msg_adv[repo].extend(rw_preds[i].tolist())

                adv_obj = {
                    "watermark": wms[i].tolist(),
                    "adv_watermark": rw_wms[i].tolist(),
                    "after_watermark": rw_instances[i].source,
                    "original_string": ori_instances[i].source,
                }
                rewatermarked_objs.append(adv_obj)

                logger.info(f"Sample No.{n_samples + 1}")
                logger.info(f"Watermark: {wms[i].tolist()}")
                logger.info(f"Predicted: {preds[i].tolist()}")
                logger.info(f"ReWatermark: {rw_wms[i].tolist()}")
                logger.info(f"ReW Predicted: {rw_preds[i].tolist()}")

                ori_str = tokens_to_strings(ori_instances[i].source_tokens)
                t_str = tokens_to_strings(t_instances[i].source_tokens)
                rw_str = tokens_to_strings(rw_instances[i].source_tokens)
                logger.info(f"Original Code: {ori_str}")
                logger.info(f"Transformed Code: {t_str}")
                logger.info(f"Updates: {updates[i]}")
                logger.info(f"ReWatermarked Code: {rw_str}")
                logger.info(f"ReWatermarked Updates: {rw_updates[i]}")

                trans_vs_vadv = calc_code_bleu.evaluate_per_example(
                    reference=t_str, hypothesis=rw_str, lang=args.lang
                )

                for key in trans_vs_vadv.keys():
                    adv_codebleu_res[key] += trans_vs_vadv[key]

                logger.info("=" * 80)

                n_samples += 1

        print(f"Number of samples: {n_samples:5d}")
        print(f"Accuracy: {tot_acc / n_samples:.4f}")
        print(f"Msg Accuracy: {tot_msg_acc / n_samples:.4f}")
        print(f"ReW Accuracy: {tot_adv_acc / n_samples:.4f}")
        print(f"ReW Msg Accuracy: {tot_adv_msg_acc / n_samples:.4f}")

        logger.info(args)
        logger.info(f"Number of samples: {n_samples:5d}")
        logger.info(f"Accuracy: {tot_acc / n_samples:.4f}")
        logger.info(f"Msg Accuracy: {tot_msg_acc / n_samples:.4f}")
        logger.info(f"ReW Accuracy: {tot_adv_acc / n_samples:.4f}")
        logger.info(f"ReW Msg Accuracy: {tot_adv_msg_acc / n_samples:.4f}")
        logger.info(f"New Accuracy: {tot_new_acc / n_samples:.4f}")
        logger.info(f"New Msg Accuracy: {tot_new_msg_acc / n_samples:.4f}")

        logger.info("CodeBLEU scores:")
        for key in adv_codebleu_res.keys():
            logger.info("Transformed vs ReWatermarking")
            logger.info(f"{key}: {adv_codebleu_res[key] / n_samples:.4f}")
            logger.info("-" * 80)

        ckpt_name = os.path.basename(os.path.dirname(args.checkpoint_path))
        if repo_wise:
            pickle.dump(
                [repowise_long_msg, repowise_long_msg_adv],
                open(f"./results/{ckpt_name}_rewater_long_{DATASET}.pkl", "wb"),
            )
        with open(f"./results/{ckpt_name}_rewater_{DATASET}.json", "w") as f:
            json.dump(rewatermarked_objs, f, indent=4)


if __name__ == "__main__":
    args = parse_args_for_evaluation()
    main(args)
