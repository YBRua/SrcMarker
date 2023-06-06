import os
import copy
import json
import torch
import random
import tree_sitter
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, RobertaTokenizer

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
from code_tokenizer import tokens_to_strings, CodeTokenizer
from eval_utils import compute_msg_acc
from code_transform_provider import CodeTransformProvider
from runtime_data_manager import InMemoryJitRuntimeDataManager
from data_processing import DataInstance
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

    parser.add_argument('--model_arch', choices=['gru', 'transformer'], default='gru')
    parser.add_argument('--shared_encoder', action='store_true')

    parser.add_argument('--write_output', action='store_true')

    return parser.parse_args()


def main(args):
    logger = setup_evaluation_logger(args, prefix='natgen')
    logger.info(args)

    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    N_BITS = args.n_bits
    CKPT_PATH = args.checkpoint_path
    DEVICE = torch.device('cuda')
    MODEL_ARCH = args.model_arch
    SHARED_ENCODER = args.shared_encoder

    # natgen
    STATE_DICT_PATH = './ckpts/checkpoint-25000/'
    natgen_tokenizer = RobertaTokenizer.from_pretrained(STATE_DICT_PATH)
    natgen = T5ForConditionalGeneration.from_pretrained(STATE_DICT_PATH)
    natgen.to(DEVICE)
    natgen.eval()

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

    dataset_processor = JsonlDatasetProcessor(LANG)

    checkpoint_dict = torch.load(CKPT_PATH, map_location='cpu')
    vocab = checkpoint_dict['vocab']
    test_instances = dataset_processor.load_jsonl(DATASET_DIR, split='test')
    raw_test_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split='test')
    test_dataset = dataset_processor.build_dataset(test_instances, vocab)
    adv_test_objs = []

    print(f'Vocab size: {len(vocab)}')
    print(f'Test size: {len(test_dataset)}')
    print(f'  Original test size: {len(test_instances)}')

    code_tokenizer = CodeTokenizer(LANG)
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
                                 vocab_mask=vocab.get_valid_identifier_mask())
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
    natgen_acc = 0

    tot_msg_acc = 0
    natgen_msg_acc = 0

    adv_codebleu_res = defaultdict(int)

    print('beginning evaluation')
    # eval starts from here
    with torch.no_grad():
        prog = tqdm(test_loader)
        for bid, batch in enumerate(prog):
            (x, lengths, src_mask, instance_ids, wms, wmids) = batch
            wms = wms.float()

            adv_obj = copy.deepcopy(raw_test_objs[bid])

            B = x.shape[0]
            x = x.to(DEVICE)
            wms = wms.to(DEVICE)
            wmids = wmids.to(DEVICE)
            src_mask = src_mask.to(DEVICE)

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

            ss_output = selector.transform_selector_forward(code_feature,
                                                            wm_feature,
                                                            transform_mask=s_masks)
            ss_ids = torch.argmax(ss_output, dim=1).tolist()

            # transform code
            ss_instances = transform_manager.get_transformed_codes_by_pred(
                instance_ids, ss_ids)
            t_instances, updates = transform_manager.varname_transform_on_instances(
                ss_instances, vs_ids)

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

            # NatGen naturalization
            # collect batched inputs
            batched_natgen_inputs = []
            for i in range(B):
                t_tokens = [t.token_value for t in t_instances[i].source_tokens]
                input_str = ' '.join(t_tokens)
                input_ids = natgen_tokenizer.encode(input_str,
                                                    max_length=512,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors='pt')
                batched_natgen_inputs.append(input_ids)
            batched_natgen_inputs = torch.cat(batched_natgen_inputs, dim=0)
            batched_natgen_inputs = batched_natgen_inputs.to(DEVICE)
            attn_mask = (batched_natgen_inputs != natgen_tokenizer.pad_token_id)

            # generate naturalized code
            natgen_preds = natgen.generate(batched_natgen_inputs,
                                           attention_mask=attn_mask,
                                           num_beams=10,
                                           max_new_tokens=1024)
            pred_nls = [
                natgen_tokenizer.decode(p, skip_special_tokens=True) for p in natgen_preds
            ]

            # build naturalized instances as inputs
            nat_instances = []
            for i in range(B):
                source_tokens, tokens = code_tokenizer.get_tokens(pred_nls[i])
                nat_instances.append(
                    DataInstance(id=t_instances[i].id,
                                 source=pred_nls[i],
                                 source_tokens=source_tokens,
                                 tokens=tokens,
                                 transform_keys=None))
                t_str = tokens_to_strings(t_instances[i].source_tokens)

            nat_x, nat_l, nat_m = transform_manager.load_to_tensor(nat_instances)
            nat_x = nat_x.to(DEVICE)
            nat_m = nat_m.to(DEVICE)

            # simulated decoding on naturalized code
            if extract_encoder is not None:
                nat_features = extract_encoder(nat_x, nat_l, nat_m)
            else:
                nat_features = encoder(nat_x, nat_l, nat_m)
            nat_outputs = wm_decoder(nat_features)
            nat_probs = torch.sigmoid(nat_outputs)
            nat_preds = (nat_probs > 0.5).long()
            natgen_acc += torch.sum(torch.mean((nat_preds == wms).float(), dim=1)).item()
            natgen_msg_acc += compute_msg_acc(nat_preds, wms, n_bits=args.n_bits)

            adv_obj['watermark'] = wms[0].tolist()
            adv_obj['extract'] = nat_preds[0].tolist()
            adv_obj['output_original_func'] = False
            adv_obj['after_watermark'] = tokens_to_strings(nat_instances[0].source_tokens)
            adv_test_objs.append(adv_obj)

            # log results
            ori_instances = transform_manager.get_original_instances(instance_ids)
            for i in range(B):
                logger.info(f'Sample No.{n_samples + 1}')
                logger.info(f'Watermark: {wms[i].tolist()}')
                logger.info(f'Predicted: {preds[i].tolist()}')
                logger.info(f'NatGen Pred: {nat_preds[i].tolist()}')

                ori_str = tokens_to_strings(ori_instances[i].source_tokens)
                t_str = tokens_to_strings(t_instances[i].source_tokens)
                nat_str = tokens_to_strings(nat_instances[i].source_tokens)
                logger.info(f'Original Code: {ori_str}')
                logger.info(f'Transformed Code: {t_str}')
                logger.info(f'Updates: {updates[i]}')
                logger.info(f'NatGen Code: {nat_str}')

                trans_vs_adv = calc_code_bleu.evaluate_per_example(reference=t_str,
                                                                   hypothesis=nat_str,
                                                                   lang=args.lang)
                for key in trans_vs_adv.keys():
                    adv_codebleu_res[key] += trans_vs_adv[key]

                logger.info('=' * 80)

                n_samples += 1

        print(f'Number of samples: {n_samples:5d}')
        print(f'Accuracy: {tot_acc / n_samples:.4f}')
        print(f'Msg Accuracy: {tot_msg_acc / n_samples:.4f}')
        print(f'NatGen Accuracy: {natgen_acc / n_samples:.4f}')
        print(f'NatGen Msg Accuracy: {natgen_msg_acc / n_samples:.4f}')

        logger.info(args)
        logger.info(f'Number of samples: {n_samples:5d}')
        logger.info(f'Accuracy: {tot_acc / n_samples:.4f}')
        logger.info(f'Msg Accuracy: {tot_msg_acc / n_samples:.4f}')
        logger.info(f'NatGen Accuracy: {natgen_acc / n_samples:.4f}')
        logger.info(f'NatGen Msg Accuracy: {natgen_msg_acc / n_samples:.4f}')
        for key in adv_codebleu_res.keys():
            logger.info('Transformed vs NatGen')
            logger.info(f'{key}: {adv_codebleu_res[key] / n_samples:.4f}')
            logger.info('-' * 80)

    assert len(test_instances) == len(adv_test_objs)
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    fo_name = f'./results/{ckpt_name}_natgen.jsonl'
    with open(fo_name, 'w') as fo:
        for instance in adv_test_objs:
            fo.write(json.dumps(instance, ensure_ascii=False) + '\n')
    print(f'Wrote adversarial examples to {fo_name}')


if __name__ == '__main__':
    args = parse_args_for_evaluation()
    main(args)
