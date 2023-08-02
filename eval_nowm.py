import os
import copy
import torch
import pickle
import random
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
        '--dataset',
        choices=['github_c_funcs', 'github_java_funcs', 'csn_java', 'csn_js'],
        default='github_c_funcs')
    parser.add_argument('--lang', choices=['cpp', 'java', 'javascript'], default='c')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/github_c_funcs')
    parser.add_argument('--n_bits', type=int, default=4)
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts/something.pt')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--model_arch', choices=['gru', 'transformer'], default='gru')
    parser.add_argument('--shared_encoder', action='store_true')
    parser.add_argument('--write_output', action='store_true')

    return parser.parse_args()


def main(args):
    logger = setup_evaluation_logger(args, prefix='nowatermark')
    logger.info(args)

    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    N_BITS = args.n_bits
    CKPT_PATH = args.checkpoint_path
    DEVICE = torch.device('cuda')
    MODEL_ARCH = args.model_arch
    SHARED_ENCODER = args.shared_encoder

    # seed
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)

    PARSER_LANG = tree_sitter.Language('parser/languages.so', args.lang)
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
        ast_transformers.VarNameStyleTransformer()
    ]

    transform_computer = CodeTransformProvider(LANG, ts_parser, code_transformers)

    dataset_processor = JsonlWMDatasetProcessor(LANG)

    checkpoint_dict = torch.load(CKPT_PATH, map_location='cpu')
    vocab = checkpoint_dict['vocab']
    test_instances = dataset_processor.load_jsonl(DATASET_DIR, split='test')
    raw_test_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split='test')
    test_dataset = dataset_processor.build_dataset(test_instances, vocab)

    print(f'Vocab size: {len(vocab)}')
    print(f'Test size: {len(test_dataset)}')
    print(f'  Original test size: {len(test_instances)}')

    valid_mask = vocab.get_valid_identifier_mask()
    print(f'  invalid mask size: {sum(valid_mask)}')
    print(f'  valid size: {len(valid_mask) - sum(valid_mask)}')

    transform_manager = InMemoryJitRuntimeDataManager(transform_computer, test_instances,
                                                      LANG)
    transform_manager.register_vocab(vocab)
    transform_manager.load_transform_mask(f'./datasets/feasible_transform_{DATASET}.json')
    transform_manager.load_varname_dict(f'./datasets/variable_names_{DATASET}.json')
    transform_capacity = transform_manager.get_transform_capacity()
    print(f'Transform capacity: {transform_capacity}')

    # build models
    print('building models')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=DynamicWMCollator(N_BITS))

    logger.info(f'Using {MODEL_ARCH}')
    if MODEL_ARCH == 'gru':
        FEATURE_DIM = 768
        encoder = GRUEncoder(vocab_size=len(vocab),
                             hidden_size=FEATURE_DIM,
                             embedding_size=FEATURE_DIM)
        if SHARED_ENCODER:
            extract_encoder = None
        else:
            extract_encoder = ExtractGRUEncoder(vocab_size=len(vocab),
                                                hidden_size=FEATURE_DIM,
                                                embedding_size=FEATURE_DIM)
    elif MODEL_ARCH == 'transformer':
        FEATURE_DIM = 768
        encoder = TransformerEncoderExtractor(vocab_size=len(vocab),
                                              embedding_size=FEATURE_DIM,
                                              hidden_size=FEATURE_DIM)
        if SHARED_ENCODER:
            extract_encoder = None
        else:
            extract_encoder = TransformerEncoderExtractor(vocab_size=len(vocab),
                                                          embedding_size=FEATURE_DIM,
                                                          hidden_size=FEATURE_DIM)

    selector = TransformSelector(vocab_size=len(vocab),
                                 transform_capacity=transform_capacity,
                                 input_dim=FEATURE_DIM,
                                 vocab_mask=vocab.get_valid_identifier_mask(),
                                 random_mask_prob=0.75)
    approximator = ConcatApproximator(vocab_size=len(vocab),
                                      transform_capacity=transform_capacity,
                                      input_dim=FEATURE_DIM,
                                      output_dim=FEATURE_DIM)
    wm_encoder = WMLinearEncoder(N_BITS, embedding_dim=FEATURE_DIM)
    wm_decoder = MLP2(output_dim=N_BITS, bn=False, input_dim=FEATURE_DIM)

    print(f'loading checkpoint from {CKPT_PATH}')
    ckpt_save = torch.load(CKPT_PATH, map_location='cpu')
    encoder.load_state_dict(ckpt_save['model'])
    if extract_encoder is not None:
        extract_encoder.load_state_dict(ckpt_save['extract_encoder'])
    wm_encoder.load_state_dict(ckpt_save['wm_encoder'])
    wm_decoder.load_state_dict(ckpt_save['wm_decoder'])
    selector.load_state_dict(ckpt_save['selector'])
    approximator.load_state_dict(ckpt_save['approximator'])

    encoder.to(DEVICE)
    if extract_encoder is not None:
        extract_encoder.to(DEVICE)
    selector.to(DEVICE)
    approximator.to(DEVICE)
    wm_encoder.to(DEVICE)
    wm_decoder.to(DEVICE)

    encoder.eval()
    if extract_encoder is not None:
        extract_encoder.eval()
    selector.eval()
    approximator.eval()
    wm_encoder.eval()
    wm_decoder.eval()

    n_samples = 0
    tot_acc = 0

    tot_msg_acc = 0

    repowise_long_msg = defaultdict(list)
    repowise_long_gt = defaultdict(list)

    print('beginning evaluation')
    # eval starts from here
    with torch.no_grad():
        prog = tqdm(test_loader)
        for bid, batch in enumerate(prog):
            test_obj = copy.deepcopy(raw_test_objs[bid])
            repo = test_obj['repo']

            (x, lengths, src_mask, instance_ids, wms, wmids) = batch
            wms = wms.float()
            B = x.shape[0]

            x = x.to(DEVICE)
            wms = wms.to(DEVICE)
            wmids = wmids.to(DEVICE)
            src_mask = src_mask.to(DEVICE)

            # use unwatermarked code as input and test the accuracy
            # should be at chance level
            code_feature = encoder(x, lengths, src_mask)
            outputs = wm_decoder(code_feature)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            tot_acc += torch.sum(torch.mean((preds == wms).float(), dim=1)).item()
            tot_msg_acc += compute_msg_acc(preds, wms, n_bits=args.n_bits)

            repowise_long_msg[repo].extend(preds[0].tolist())
            repowise_long_gt[repo].extend(wms[0].long().tolist())

            # log results
            ori_instances = transform_manager.get_original_instances(instance_ids)
            for i in range(B):
                logger.info(f'Sample No.{n_samples + 1}')
                logger.info(f'Watermark: {wms[i].long().tolist()}')
                logger.info(f'Predicted: {preds[i].tolist()}')

                ori_str = tokens_to_strings(ori_instances[i].source_tokens)
                logger.info(f'Original Code: {ori_str}')
                logger.info('=' * 80)

                n_samples += 1

        print(f'Number of samples: {n_samples:5d}')
        print(f'Accuracy: {tot_acc / n_samples:.4f}')
        print(f'Msg Accuracy: {tot_msg_acc / n_samples:.4f}')

        logger.info(args)
        logger.info(f'Number of samples: {n_samples:5d}')
        logger.info(f'Accuracy: {tot_acc / n_samples:.4f}')
        logger.info(f'Msg Accuracy: {tot_msg_acc / n_samples:.4f}')

    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint_path))

    if args.write_output:
        pickle.dump([repowise_long_msg, repowise_long_gt],
                    open(f'./results/{ckpt_name}_long_nowm.pkl', 'wb'))


if __name__ == '__main__':
    args = parse_args_for_evaluation()
    main(args)
