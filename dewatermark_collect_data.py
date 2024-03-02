import os
import copy
import json
import torch
import random
import tree_sitter
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from models import (
    TransformSelector,
    TransformerEncoderExtractor,
    GRUEncoder,
    WMLinearEncoder,
)

from code_transform_provider import CodeTransformProvider
from runtime_data_manager import InMemoryJitRuntimeDataManager
from data_processing import (
    DataInstance,
    CodeVocab,
    JsonlWMDatasetProcessor,
    DynamicWMCollator,
)
import mutable_tree.transformers as ast_transformers
from typing import List


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
        ],
        default="github_c_funcs",
    )
    parser.add_argument("--lang", choices=["cpp", "java", "javascript"], default="cpp")
    parser.add_argument("--dataset_dir", type=str, default="./datasets/github_c_funcs")
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--checkpoint_path", type=str, default="./ckpts/something.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--varmask_prob", type=float, default=0.5)
    parser.add_argument("--model_arch", choices=["gru", "transformer"], default="gru")
    parser.add_argument(
        "--var_transform_mode", choices=["replace", "append"], default="replace"
    )

    return parser.parse_args()


def gather_samples(
    instances: List[DataInstance],
    objs: List,
    vocab: CodeVocab,
    dataset_processor: JsonlWMDatasetProcessor,
    transform_manager: InMemoryJitRuntimeDataManager,
    encoder: nn.Module,
    wm_encoder: WMLinearEncoder,
    selector: TransformSelector,
    device: torch.device,
    var_transform_mode: str = "replace",
):
    dewatermark_samples = []
    dataset = dataset_processor.build_dataset(instances, vocab)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=DynamicWMCollator(4)
    )

    transform_capacity = transform_manager.get_transform_capacity()
    with torch.no_grad():
        prog = tqdm(dataloader)
        for bid, batch in enumerate(prog):
            obj = copy.deepcopy(objs[bid])

            (x, lengths, src_mask, instance_ids, wms, wmids) = batch
            x = x.to(device)
            wms = wms.to(device)
            wmids = wmids.to(device)
            src_mask = src_mask.to(device)

            wms = wms.float()

            # get style masks
            s_feasible = transform_manager.get_feasible_transform_ids(instance_ids)
            s_feasible_01 = []
            for s_f in s_feasible:
                val = torch.ones(transform_capacity, device=device).bool()
                val[s_f] = False
                s_feasible_01.append(val)
            s_masks = torch.stack(s_feasible_01, dim=0)

            # simulated watermark encoding
            # get transform and variable substitution predictions
            code_feature = encoder(x, lengths, src_mask)
            wm_feature = wm_encoder(wms)

            vs_output = selector.var_selector_forward(
                code_feature, wm_feature, random_mask=True
            )
            vs_ids = torch.argmax(vs_output, dim=1).tolist()

            ss_output = selector.transform_selector_forward(
                code_feature, wm_feature, transform_mask=s_masks
            )
            ss_ids = torch.argmax(ss_output, dim=1).tolist()

            # transform code
            ori_instances = transform_manager.get_original_instances(instance_ids)
            t_instances, updates = transform_manager.varname_transform_on_instances(
                ori_instances, vs_ids, mode=var_transform_mode
            )
            t_instances = transform_manager.transform_on_instances(t_instances, ss_ids)

            orig_instances = transform_manager.get_original_instances(instance_ids)

            assert len(t_instances) == 1
            dewatermark_obj = {
                "original_string": obj["original_string"],
                "original_tokens": orig_instances[0].tokens,
                "after_watermark": t_instances[0].source,
                "after_watermark_tokens": t_instances[0].tokens,
                "contains_watermark": True,
                "docstring_tokens": obj["docstring_tokens"]
                if "docstring_tokens" in obj
                else None,
                "watermark": wms[0].long().tolist(),
            }
            dewatermark_samples.append(dewatermark_obj)

    assert len(dewatermark_samples) == len(instances)
    return dewatermark_samples


def main(args):
    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    N_BITS = args.n_bits
    CKPT_PATH = args.checkpoint_path
    DEVICE = torch.device("cuda")
    MODEL_ARCH = args.model_arch
    VARMASK_PROB = args.varmask_prob
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
    vocab: CodeVocab = checkpoint_dict["vocab"]

    if DATASET in {"mbjp", "mbjsp"}:
        test_instances = dataset_processor.load_jsonl(DATASET_DIR, split="test")
        raw_test_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split="test")
        all_instances = test_instances
    else:
        train_instances = dataset_processor.load_jsonl(DATASET_DIR, split="train")
        valid_instances = dataset_processor.load_jsonl(DATASET_DIR, split="valid")
        test_instances = dataset_processor.load_jsonl(DATASET_DIR, split="test")
        raw_train_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split="train")
        raw_valid_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split="valid")
        raw_test_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split="test")

        all_instances = train_instances + valid_instances + test_instances

    transform_manager = InMemoryJitRuntimeDataManager(
        transform_computer, all_instances, LANG
    )
    transform_manager.register_vocab(vocab)
    transform_manager.load_transform_mask(
        f"./datasets/feasible_transform_{DATASET}.json"
    )
    transform_manager.load_varname_dict(f"./datasets/variable_names_{DATASET}.json")
    transform_capacity = transform_manager.get_transform_capacity()
    print(f"Transform capacity: {transform_capacity}")

    # build models
    if VAR_TRANSFORM_MODE == "replace":
        vmask = vocab.get_valid_identifier_mask()
    else:
        vmask = vocab.get_valid_highfreq_mask(2**N_BITS * 32)

    print("building models")
    print(f"Using {MODEL_ARCH}")
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

    selector = TransformSelector(
        vocab_size=len(vocab),
        transform_capacity=transform_capacity,
        input_dim=FEATURE_DIM,
        vocab_mask=vmask,
        random_mask_prob=VARMASK_PROB,
    )
    wm_encoder = WMLinearEncoder(N_BITS, embedding_dim=FEATURE_DIM)

    print(f"loading checkpoint from {CKPT_PATH}")
    ckpt_save = torch.load(CKPT_PATH, map_location="cpu")
    encoder.load_state_dict(ckpt_save["model"])
    wm_encoder.load_state_dict(ckpt_save["wm_encoder"])
    selector.load_state_dict(ckpt_save["selector"])

    encoder.to(DEVICE)
    selector.to(DEVICE)
    wm_encoder.to(DEVICE)

    encoder.eval()
    selector.eval()
    wm_encoder.eval()

    if DATASET in {"mbjp", "mbjsp"}:
        detector_test_samples = gather_samples(
            test_instances,
            raw_test_objs,
            vocab,
            dataset_processor,
            transform_manager,
            encoder,
            wm_encoder,
            selector,
            DEVICE,
            VAR_TRANSFORM_MODE,
        )
    else:
        detector_train_samples = gather_samples(
            train_instances,
            raw_train_objs,
            vocab,
            dataset_processor,
            transform_manager,
            encoder,
            wm_encoder,
            selector,
            DEVICE,
            VAR_TRANSFORM_MODE,
        )
        detector_valid_samples = gather_samples(
            valid_instances,
            raw_valid_objs,
            vocab,
            dataset_processor,
            transform_manager,
            encoder,
            wm_encoder,
            selector,
            DEVICE,
            VAR_TRANSFORM_MODE,
        )
        detector_test_samples = gather_samples(
            test_instances,
            raw_test_objs,
            vocab,
            dataset_processor,
            transform_manager,
            encoder,
            wm_encoder,
            selector,
            DEVICE,
            VAR_TRANSFORM_MODE,
        )

    ckpt_name = os.path.basename(os.path.dirname(CKPT_PATH))
    DATASET_SAVE_DIR = f"./datasets/dewatermark/{DATASET}/{ckpt_name}"
    if not os.path.exists(DATASET_SAVE_DIR):
        os.makedirs(DATASET_SAVE_DIR)

    if DATASET not in {"mbjp", "mbjsp"}:
        with open(os.path.join(DATASET_SAVE_DIR, "train.jsonl"), "w") as f:
            for sample in detector_train_samples:
                f.write(json.dumps(sample) + "\n")

        with open(os.path.join(DATASET_SAVE_DIR, "valid.jsonl"), "w") as f:
            for sample in detector_valid_samples:
                f.write(json.dumps(sample) + "\n")

    with open(os.path.join(DATASET_SAVE_DIR, "test.jsonl"), "w") as f:
        for sample in detector_test_samples:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    args = parse_args_for_evaluation()
    main(args)
