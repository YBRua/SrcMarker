import os
import copy
import json
import time
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

from metrics import calc_code_bleu
from metrics.syntax_match import check_tree_validity
from code_tokenizer import tokens_to_strings
from eval_utils import JitAdversarialTransformProvider, compute_msg_acc
from code_transform_provider import CodeTransformProvider
from runtime_data_manager import InMemoryJitRuntimeDataManager
from data_processing import JsonlDatasetProcessor, DynamicWMCollator
from logger_setup import setup_evaluation_logger
import mutable_tree.transformers as ast_transformers


def parse_args_for_evaluation():
    parser = ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['github_c_funcs', 'github_java_funcs', 'csn_java'],
                        default='github_c_funcs')
    parser.add_argument('--lang', choices=['cpp', 'java'], default='c')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/github_c_funcs')
    parser.add_argument('--n_bits', type=int, default=4)
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts/something.pt')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--trans_adv', action='store_true')
    parser.add_argument('--n_trans_adv', type=int, default=1)

    parser.add_argument('--var_adv', action='store_true')
    parser.add_argument('--var_nomask', action='store_true')
    parser.add_argument('--var_adv_proportion', type=float, default=None)
    parser.add_argument('--var_adv_budget', type=int, default=None)

    parser.add_argument('--model_arch', choices=['gru', 'transformer'], default='gru')
    parser.add_argument('--shared_encoder', action='store_true')

    parser.add_argument('--all_adv', action='store_true')
    parser.add_argument('--write_output', action='store_true')

    return parser.parse_args()


def main(args):
    logger = setup_evaluation_logger(args)
    logger.info(args)

    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    N_BITS = args.n_bits
    CKPT_PATH = args.checkpoint_path
    DEVICE = torch.device('cuda')
    MODEL_ARCH = args.model_arch
    SHARED_ENCODER = args.shared_encoder
    RANDOM_MASK = not args.var_nomask

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
        ast_transformers.UpdateTransformer(),
        ast_transformers.SameTypeDeclarationTransformer(),
        ast_transformers.VarDeclLocationTransformer(),
        ast_transformers.VarInitTransformer(),
        ast_transformers.VarNameStyleTransformer()
    ]

    transform_computer = CodeTransformProvider(LANG, ts_parser, code_transformers)
    adv = JitAdversarialTransformProvider(
        transform_computer,
        transforms_per_file=f'./datasets/transforms_per_file_{DATASET}.json',
        varname_path=f'./datasets/variable_names_{DATASET}.json',
        lang=LANG)

    dataset_processor = JsonlDatasetProcessor(LANG)

    checkpoint_dict = torch.load(CKPT_PATH, map_location='cpu')
    vocab = checkpoint_dict['vocab']
    test_instances = dataset_processor.load_jsonl(DATASET_DIR, split='test')
    raw_test_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split='test')
    test_dataset = dataset_processor.build_dataset(test_instances, vocab)
    new_test_objs = []
    adv_test_objs = []

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
    sadv_acc = 0
    vadv_acc = 0
    dcadv_acc = 0

    tot_msg_acc = 0
    sadv_msg_acc = 0
    vadv_msg_acc = 0
    dcadv_msg_acc = 0

    tot_embed_time = 0
    tot_extract_time = 0

    codebleu_res = defaultdict(int)
    adv_codebleu_res = defaultdict(int)
    repowise_long_msg = defaultdict(list)
    repowise_long_gt = defaultdict(list)

    valid_t = 0
    valid_all = len(test_dataset)

    print('beginning evaluation')
    # eval starts from here
    with torch.no_grad():
        prog = tqdm(test_loader)
        for bid, batch in enumerate(prog):
            test_obj = copy.deepcopy(raw_test_objs[bid])
            repo = test_obj['repo']

            adv_obj = copy.deepcopy(raw_test_objs[bid])

            (x, lengths, src_mask, instance_ids, wms, wmids) = batch
            wms = wms.float()
            B = x.shape[0]

            x = x.to(DEVICE)
            wms = wms.to(DEVICE)
            wmids = wmids.to(DEVICE)
            src_mask = src_mask.to(DEVICE)

            # get style masks
            embed_start = time.time()
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

            vs_output = selector.var_selector_forward(code_feature,
                                                      wm_feature,
                                                      random_mask=RANDOM_MASK)
            vs_ids = torch.argmax(vs_output, dim=1).tolist()

            ss_output = selector.transform_selector_forward(code_feature,
                                                            wm_feature,
                                                            transform_mask=s_masks)
            ss_ids = torch.argmax(ss_output, dim=1).tolist()

            # transform code
            ss_instances = transform_manager.get_transformed_codes_by_pred(
                instance_ids, ss_ids)
            t_instances, updates = transform_manager.varname_transform_on_instances(
                ss_instances, vs_ids)
            embed_time = time.time() - embed_start

            # simulated decoding process
            dec_x, dec_l, dec_m = transform_manager.load_to_tensor(t_instances)
            dec_x = dec_x.to(DEVICE)
            dec_m = dec_m.to(DEVICE)

            extract_start = time.time()
            if extract_encoder is not None:
                t_features = extract_encoder(dec_x, dec_l, dec_m)
            else:
                t_features = encoder(dec_x, dec_l, dec_m)
            outputs = wm_decoder(t_features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
            extract_time = time.time() - extract_start

            tot_embed_time += embed_time
            tot_extract_time += extract_time

            tot_acc += torch.sum(torch.mean((preds == wms).float(), dim=1)).item()
            tot_msg_acc += compute_msg_acc(preds, wms, n_bits=args.n_bits)

            repowise_long_msg[repo].extend(preds[0].tolist())
            repowise_long_gt[repo].extend(wms[0].long().tolist())

            # update instance
            test_obj['watermark'] = wms[0].long().tolist()
            test_obj['extract'] = preds[0].tolist()
            test_obj['output_original_func'] = False
            test_obj['after_watermark'] = tokens_to_strings(t_instances[0].source_tokens)
            new_test_objs.append(test_obj)

            # code style adversarial attack
            if args.trans_adv:
                # sadv_ids = torch.randint(0, transform_capacity, (B, ), device=DEVICE)
                sadv_ids = adv.get_adv_style_transforms(t_instances, args.n_trans_adv)
                sadv_instances = transform_manager.get_transformed_codes_by_pred(
                    instance_ids, sadv_ids)
                # recover substituted variable names
                sadv_instances = adv.redo_varname_transform(sadv_instances, updates)
                sadv_x, sadv_l, sadv_m = transform_manager.load_to_tensor(sadv_instances)
                sadv_x = sadv_x.to(DEVICE)
                sadv_m = sadv_m.to(DEVICE)

                if extract_encoder is not None:
                    sadv_features = extract_encoder(sadv_x, sadv_l, sadv_m)
                else:
                    sadv_features = encoder(sadv_x, sadv_l, sadv_m)

                sadv_outputs = wm_decoder(sadv_features)
                sadv_probs = torch.sigmoid(sadv_outputs)
                sadv_preds = (sadv_probs > 0.5).long()
                sadv_acc += torch.sum(torch.mean((sadv_preds == wms).float(),
                                                 dim=1)).item()
                sadv_msg_acc += compute_msg_acc(sadv_preds, wms, n_bits=args.n_bits)

                adv_obj['watermark'] = wms[0].long().tolist()
                adv_obj['extract'] = sadv_preds[0].tolist()
                adv_obj['output_original_func'] = False
                adv_obj['after_watermark'] = tokens_to_strings(
                    sadv_instances[0].source_tokens)
                adv_test_objs.append(adv_obj)

            # variable substitution attack
            if args.var_adv:
                vadv_instances, vadv_updates = adv.adv_varname_transform(
                    t_instances, proportion=args.var_adv_proportion, var_updates=updates)
                vadv_x, vadv_l, vadv_m = transform_manager.load_to_tensor(vadv_instances)
                vadv_x = vadv_x.to(DEVICE)
                vadv_m = vadv_m.to(DEVICE)

                if extract_encoder is not None:
                    vadv_features = extract_encoder(vadv_x, vadv_l, vadv_m)
                else:
                    vadv_features = encoder(vadv_x, vadv_l, vadv_m)

                vadv_outputs = wm_decoder(vadv_features)
                vadv_probs = torch.sigmoid(vadv_outputs)
                vadv_preds = (vadv_probs > 0.5).long()
                vadv_acc += torch.sum(torch.mean((vadv_preds == wms).float(),
                                                 dim=1)).item()
                vadv_msg_acc += compute_msg_acc(vadv_preds, wms, n_bits=args.n_bits)

                adv_obj['watermark'] = wms[0].long().tolist()
                adv_obj['extract'] = vadv_preds[0].tolist()
                adv_obj['output_original_func'] = False
                adv_obj['after_watermark'] = tokens_to_strings(
                    vadv_instances[0].source_tokens)
                adv_test_objs.append(adv_obj)

            if args.all_adv:
                # dual-channel adversary
                # fixed 50% random substitution & 2 random transformations
                dcadv_tids = adv.get_adv_style_transforms(t_instances, n_transforms=2)
                dcadv_instances = transform_manager.get_transformed_codes_by_pred(
                    instance_ids, dcadv_tids)
                # recover substituted variable names
                dcadv_instances = adv.redo_varname_transform(dcadv_instances, updates)
                dcadv_instances, dcadv_updates = adv.adv_varname_transform(
                    dcadv_instances, proportion=0.5, var_updates=updates)

                dcadv_x, dcadv_l, dcadv_m = transform_manager.load_to_tensor(
                    dcadv_instances)
                dcadv_x = dcadv_x.to(DEVICE)
                dcadv_m = dcadv_m.to(DEVICE)

                if extract_encoder is not None:
                    dcadv_features = extract_encoder(dcadv_x, dcadv_l, dcadv_m)
                else:
                    dcadv_features = encoder(dcadv_x, dcadv_l, dcadv_m)

                dcadv_outputs = wm_decoder(dcadv_features)
                dcadv_probs = torch.sigmoid(dcadv_outputs)
                dcadv_preds = (dcadv_probs > 0.5).long()
                dcadv_acc += torch.sum(torch.mean((dcadv_preds == wms).float(),
                                                  dim=1)).item()
                dcadv_msg_acc += compute_msg_acc(dcadv_preds, wms, n_bits=args.n_bits)

                adv_obj['watermark'] = wms[0].long().tolist()
                adv_obj['extract'] = dcadv_preds[0].tolist()
                adv_obj['output_original_func'] = False
                adv_obj['after_watermark'] = tokens_to_strings(
                    dcadv_instances[0].source_tokens)
                adv_test_objs.append(adv_obj)

            # log results
            ori_instances = transform_manager.get_original_instances(instance_ids)
            for i in range(B):
                logger.info(f'Sample No.{n_samples + 1}')
                logger.info(f'Watermark: {wms[i].long().tolist()}')
                logger.info(f'Predicted: {preds[i].tolist()}')
                if args.trans_adv:
                    logger.info(f'Style Adv Pred: {sadv_preds[i].tolist()}')
                if args.var_adv:
                    logger.info(f'Var Adv Pred: {vadv_preds[i].tolist()}')

                ori_str = tokens_to_strings(ori_instances[i].source_tokens)
                t_str = tokens_to_strings(t_instances[i].source_tokens)
                logger.info(f'Original Code: {ori_str}')
                logger.info(f'Transformed Code: {t_str}')
                logger.info(f'Variable Updates: {updates[i]}')
                transform_keys = transform_computer.get_transform_keys()[ss_ids[i]]
                logger.info(f'Style Changes: {transform_keys}')

                ori_vs_trans = calc_code_bleu.evaluate_per_example(ori_str,
                                                                   t_str,
                                                                   lang=LANG)
                for key in ori_vs_trans.keys():
                    codebleu_res[key] += ori_vs_trans[key]

                # NOTE: we are using the orinal source code to check the validity
                # because the string reconstructed from tokens is not lossless
                if LANG == 'java':
                    t_source = f'public class Wrapper {{ {t_instances[i].source} }}'
                else:
                    t_source = t_instances[i].source
                if check_tree_validity(
                        ts_parser.parse(t_source.encode('utf-8')).root_node):
                    valid_t += 1
                else:
                    print(f'{n_samples + 1} invalid (transformed);')
                    logger.info('!!! Invalid Transformed Code !!!')
                    logger.info(t_instances[i].id)
                    logger.info(t_instances[i].transform_keys)
                    logger.info(t_instances[i].source)
                    logger.info('!!! End of Sample !!!')

                if args.var_adv:
                    vadv_str = tokens_to_strings(vadv_instances[i].source_tokens)
                    logger.info(f'Var Adv Code: {vadv_str}')
                    logger.info(f'Var Adv Updates: {vadv_updates[i]}')

                    trans_vs_adv = calc_code_bleu.evaluate_per_example(
                        reference=t_str, hypothesis=vadv_str, lang=args.lang)
                    for key in trans_vs_adv.keys():
                        adv_codebleu_res[key] += trans_vs_adv[key]

                if args.trans_adv:
                    sadv_str = tokens_to_strings(sadv_instances[i].source_tokens)
                    logger.info(f'Style Adv Code: {sadv_str}')
                    sadv_keys = transform_manager.get_transform_keys()[sadv_ids[i]]
                    logger.info(f'Style Adv Changes: {sadv_keys}')

                    trans_vs_adv = calc_code_bleu.evaluate_per_example(
                        reference=t_str, hypothesis=sadv_str, lang=args.lang)
                    for key in trans_vs_adv.keys():
                        adv_codebleu_res[key] += trans_vs_adv[key]

                if args.all_adv:
                    dcadv_str = tokens_to_strings(dcadv_instances[i].source_tokens)
                    logger.info(f'Dual Adv Code: {dcadv_str}')
                    dcadv_keys = transform_manager.get_transform_keys()[dcadv_tids[i]]
                    logger.info(f'Dual Adv Changes: {dcadv_keys}')
                    logger.info(f'Dual Adv Updates: {dcadv_updates[i]}')

                    trans_vs_adv = calc_code_bleu.evaluate_per_example(
                        reference=t_str, hypothesis=dcadv_str, lang=args.lang)
                    for key in trans_vs_adv.keys():
                        adv_codebleu_res[key] += trans_vs_adv[key]

                logger.info('=' * 80)

                n_samples += 1

        print(f'Number of samples: {n_samples:5d}')
        print(f'Accuracy: {tot_acc / n_samples:.4f}')
        print(f'Msg Accuracy: {tot_msg_acc / n_samples:.4f}')
        if args.trans_adv:
            print(f'Style Adv Accuracy: {sadv_acc / n_samples:.4f}')
            print(f'Style Adv Msg Accuracy: {sadv_msg_acc / n_samples:.4f}')
        if args.var_adv:
            print(f'Var Adv Accuracy: {vadv_acc / n_samples:.4f}')
            print(f'Var Adv Msg Accuracy: {vadv_msg_acc / n_samples:.4f}')
        if args.all_adv:
            print(f'Dual Adv Accuracy: {dcadv_acc / n_samples:.4f}')
            print(f'Dual Adv Msg Accuracy: {dcadv_msg_acc / n_samples:.4f}')

        logger.info(args)
        logger.info(f'Number of samples: {n_samples:5d}')
        logger.info(f'Accuracy: {tot_acc / n_samples:.4f}')
        logger.info(f'Msg Accuracy: {tot_msg_acc / n_samples:.4f}')
        if args.trans_adv:
            logger.info(f'Style Adv Accuracy: {sadv_acc / n_samples:.4f}')
            logger.info(f'Style Adv Msg Accuracy: {sadv_msg_acc / n_samples:.4f}')
        if args.var_adv:
            logger.info(f'Var Adv Accuracy: {vadv_acc / n_samples:.4f}')
            logger.info(f'Var Adv Msg Accuracy: {vadv_msg_acc / n_samples:.4f}')
        if args.all_adv:
            logger.info(f'Dual Adv Accuracy: {dcadv_acc / n_samples:.4f}')
            logger.info(f'Dual Adv Msg Accuracy: {dcadv_msg_acc / n_samples:.4f}')

        logger.info('CodeBLEU scores:')
        for key in codebleu_res.keys():
            logger.info('Original vs Transformed')
            logger.info(f'{key}: {codebleu_res[key] / n_samples:.4f}')
            logger.info('-' * 80)

        if args.trans_adv or args.var_adv or args.all_adv:
            for key in adv_codebleu_res.keys():
                logger.info('Transformed vs Adv')
                logger.info(f'{key}: {adv_codebleu_res[key] / n_samples:.4f}')
                logger.info('-' * 80)

        logger.info(
            f'Valid transformed: {valid_t}/{valid_all} ({valid_t / valid_all:.4f})')

        logger.info(f'Average Embedding Time: {tot_embed_time / n_samples:.4f}')
        logger.info(f'Average Extraction Time: {tot_extract_time / n_samples:.4f}')
        avg_total_time = tot_embed_time / n_samples + tot_extract_time / n_samples
        logger.info(f'Average Total Time: {avg_total_time:.4f}')

    assert len(test_instances) == len(new_test_objs)
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    if args.write_output:
        fo_name = f'./results/{ckpt_name}_test.jsonl'
        with open(fo_name, 'w') as fo:
            for instance in new_test_objs:
                fo.write(json.dumps(instance, ensure_ascii=False) + '\n')
        print(f'Wrote test examples to {fo_name}')

    if args.var_adv or args.trans_adv or args.all_adv:
        assert len(test_instances) == len(adv_test_objs)
        if args.var_adv:
            label = (int(args.var_adv_proportion * 100)
                     if args.var_adv_proportion is not None else args.var_adv_budget)
            fo_name = f'./results/{ckpt_name}_vadv{label}.jsonl'
        if args.trans_adv:
            fo_name = f'./results/{ckpt_name}_tadv_{args.n_trans_adv}.jsonl'
        if args.all_adv:
            fo_name = f'./results/{ckpt_name}_dcadv.jsonl'
        with open(fo_name, 'w') as fo:
            for instance in adv_test_objs:
                fo.write(json.dumps(instance, ensure_ascii=False) + '\n')
        print(f'Wrote adversarial examples to {fo_name}')

    if args.write_output:
        pickle.dump([repowise_long_msg, repowise_long_gt],
                    open(f'./results/{ckpt_name}_long.pkl', 'wb'))


if __name__ == '__main__':
    args = parse_args_for_evaluation()
    main(args)
