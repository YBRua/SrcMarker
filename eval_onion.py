import os
import copy
import math
import json
import time
import torch
import random
import tree_sitter
import transformers
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sctokenizer import Token, TokenType

from models import (
    ConcatApproximator,
    TransformSelector,
    TransformerEncoderExtractor,
    GRUEncoder,
    ExtractGRUEncoder,
    WMLinearEncoder,
    MLP2,
)

import mutable_tree
from metrics import calc_code_bleu
from data_processing import DataInstance
from code_tokenizer import tokens_to_strings, CodeTokenizer
from eval_utils import JitAdversarialTransformProvider, compute_msg_acc
from code_transform_provider import CodeTransformProvider
from runtime_data_manager import InMemoryJitRuntimeDataManager
from data_processing import JsonlDatasetProcessor, DynamicWMCollator
from logger_setup import setup_evaluation_logger
import mutable_tree.transformers as ast_transformers
from varname_utils import normalize_name
from typing import List, Set, Tuple


class GPT2LM:
    def __init__(self, device=None):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            "microsoft/CodeGPT-small-java")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained(
            "microsoft/CodeGPT-small-java", from_tf=False)
        self.lm.to(device)

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        ipt = self.tokenizer(
            sent,
            return_tensors="pt",
            verbose=False,
        )
        if any(len(x) > 512 for x in ipt['input_ids']):
            return np.nan
        try:
            ppl = math.exp(
                self.lm(input_ids=ipt['input_ids'].cuda(),
                        attention_mask=ipt['attention_mask'].cuda(),
                        labels=ipt.input_ids.cuda())[0])
        except RuntimeError:
            ppl = np.nan
        return ppl


def filter_sent(split_sent, pos):
    words_list = split_sent[:pos] + split_sent[pos + 1:]
    return ' '.join(words_list)


def get_perplexity(toks: List[Token], gpt_lm) -> List[float]:
    split_sent = [tok.token_value for tok in toks]
    sent_length = len(split_sent)
    single_sent_ppls = []
    for j in range(sent_length):
        processed_sent = filter_sent(split_sent, j)
        single_sent_ppls.append(gpt_lm(processed_sent))

    return single_sent_ppls


def onion_detect(instance: DataInstance, sent_ppls: List[str], thres: float,
                 update: Tuple[str, str], keywords: Set[str], vars: List[str],
                 whitelist: Set[str]):
    assert thres < 0
    new_instance = copy.deepcopy(instance)
    source_tokens = new_instance.source_tokens
    sent_split = [tok.token_value for tok in instance.source_tokens][:-1]
    assert len(sent_split) == len(sent_ppls) - 1

    whole_sent_ppl = sent_ppls[-1]
    diffs = [ppl - whole_sent_ppl for ppl in sent_ppls][:-1]
    suspects = [ppl_gap <= thres for ppl_gap in diffs]

    tp = 0
    fp = 0
    fn = 0
    hit = False
    var_map = {}
    suspect_strs = []
    for i, is_suspect in enumerate(suspects):
        if is_suspect:
            suspect_str = sent_split[i]
            suspect_tok = source_tokens[i]
            if normalize_name(suspect_str) not in whitelist:
                continue
            if (suspect_str.isidentifier() and suspect_str not in keywords
                    and suspect_tok.token_type == TokenType.IDENTIFIER):
                if suspect_str not in var_map:
                    var_map[suspect_str] = random.choice(vars)

                # replace all occurrences
                for tok in new_instance.source_tokens:
                    if tok.token_value == suspect_str:
                        tok.token_value = var_map[suspect_str]

            if suspect_str == update[1] and not hit:
                tp += 1
                hit = True
            elif suspect_str != update[1]:
                fp += 1
            suspect_strs.append(suspect_str)

    if not hit:
        fn += 1

    return suspect_strs, new_instance, tp, fp, fn


def get_function_name(root: mutable_tree.nodes.FunctionDeclaration):
    declarator = root.header.func_decl

    def _get_identifier_from_decl(node: mutable_tree.nodes.Declarator):
        if isinstance(node, mutable_tree.nodes.VariableDeclarator):
            return node.decl_id.name
        else:
            return _get_identifier_from_decl(node.declarator)

    return _get_identifier_from_decl(declarator)


if __name__ == '__main__':
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
    args = parser.parse_args()

    logger = setup_evaluation_logger(args, prefix='onion')
    logger.info(args)

    LANG = args.lang
    DATASET = args.dataset
    DATASET_DIR = args.dataset_dir
    N_BITS = args.n_bits
    CKPT_PATH = args.checkpoint_path
    MODEL_ARCH = args.model_arch
    SHARED_ENCODER = args.shared_encoder
    DEVICE = torch.device('cuda')

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
    all_var_names = adv.all_varnames
    varnames_per_file = adv.varnames_per_file

    dataset_processor = JsonlDatasetProcessor(LANG)
    code_tokenizer = CodeTokenizer(LANG)
    print('loading original dataset')
    checkpoint_dict = torch.load(CKPT_PATH, map_location='cpu')
    vocab = checkpoint_dict['vocab']
    test_instances = dataset_processor.load_jsonl(DATASET_DIR, split='test')
    raw_test_objs = dataset_processor.load_raw_jsonl(DATASET_DIR, split='test')
    test_dataset = dataset_processor.build_dataset(test_instances, vocab)
    adv_test_objs = []

    print(f'Test size: {len(test_dataset)}')

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

    # ONION models and keywords
    print('building language model')
    gpt_lm = GPT2LM(device=DEVICE)
    # load keywords
    KW_LANG = 'c' if LANG == 'cpp' else LANG
    KW_PATH = f'./metrics/keywords/{KW_LANG}.txt'
    with open(KW_PATH, 'r') as f:
        keywords = set([kw.strip() for kw in f.readlines()])

    # evaluation
    tot_acc = 0
    tot_msg_acc = 0
    tot_detect_f1 = 0
    tot_detect_tpr = 0
    tot_detect_fpr = 0
    adv_acc = 0
    adv_msg_acc = 0
    n_samples = 0
    adv_codebleu_res = defaultdict(int)

    print('beginning evaluation')
    # eval starts from here
    with torch.no_grad():
        pairs = []
        prog = tqdm(test_loader)
        for bid, batch in enumerate(prog):
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

            extract_start = time.time()
            if extract_encoder is not None:
                t_features = extract_encoder(dec_x, dec_l, dec_m)
            else:
                t_features = encoder(dec_x, dec_l, dec_m)
            outputs = wm_decoder(t_features)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            tot_acc += torch.sum(torch.mean((preds == wms).float(), dim=1)).item()
            tot_msg_acc += compute_msg_acc(preds, wms, n_bits=args.n_bits)

            # ONION trigger word detection and removal
            toks = t_instances[0].source_tokens
            tok_strs = [tok.token_value for tok in toks]
            sent_ppl = get_perplexity(toks, gpt_lm)
            pairs.append((tok_strs, sent_ppl, updates[0]))

            # only replace variables
            whitelist = set(
                normalize_name(x) for x in varnames_per_file[t_instances[0].id])
            whitelist.add(updates[0][1])

            suspect_strs, new_instance, tp, fp, fn = onion_detect(t_instances[0],
                                                                  sent_ppl,
                                                                  thres=-2,
                                                                  update=updates[0],
                                                                  keywords=keywords,
                                                                  vars=all_var_names,
                                                                  whitelist=whitelist)
            new_instance.tokens = dataset_processor.code_tokenizer._tokens_postprocess(
                new_instance.source_tokens)

            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
            tpr = tp / (tp + fn) if tp + fn > 0 else 0
            tot_detect_f1 += f1
            tot_detect_tpr += tpr

            x_adv, l_adv, m_adv = transform_manager.load_to_tensor([new_instance])
            x_adv = x_adv.to(DEVICE)
            m_adv = m_adv.to(DEVICE)

            if extract_encoder is not None:
                adv_features = extract_encoder(x_adv, l_adv, m_adv)
            else:
                adv_features = encoder(x_adv, l_adv, m_adv)
            adv_outputs = wm_decoder(adv_features)
            adv_probs = torch.sigmoid(adv_outputs)
            adv_preds = (adv_probs > 0.5).long()
            adv_acc += torch.sum(torch.mean((adv_preds == wms).float(), dim=1)).item()
            adv_msg_acc += compute_msg_acc(adv_preds, wms, n_bits=args.n_bits)

            ori_instances = transform_manager.get_original_instances(instance_ids)
            for i in range(B):
                n_samples += 1
                logger.info(f'Sample No.{n_samples}')
                logger.info(f'Watermark: {wms[i].tolist()}')
                logger.info(f'Predicted: {preds[i].tolist()}')
                logger.info(f'ONION Adv Pred: {adv_preds[i].tolist()}')
                ori_str = tokens_to_strings(ori_instances[i].source_tokens)
                t_str = tokens_to_strings(t_instances[i].source_tokens)
                adv_str = tokens_to_strings(new_instance.source_tokens)
                logger.info(f'Original Code: {ori_str}')
                logger.info(f'Transformed Code: {t_str}')
                logger.info(f'Updates: {updates[i]}')
                logger.info(f'Suspects: {suspect_strs}')
                logger.info(f'Whitelist: {whitelist}')
                logger.info(f'ONION Code: {adv_str}')

                adv_obj['watermark'] = wms[0].tolist()
                adv_obj['extract'] = adv_preds[0].tolist()
                adv_obj['output_original_func'] = False
                adv_obj['after_watermark'] = adv_str
                adv_test_objs.append(adv_obj)

                trans_vs_adv = calc_code_bleu.evaluate_per_example(reference=t_str,
                                                                   hypothesis=adv_str,
                                                                   lang=args.lang)
                for key in trans_vs_adv.keys():
                    adv_codebleu_res[key] += trans_vs_adv[key]

                logger.info('=' * 80)

    logger.info(args)
    print(f'Number of samples: {n_samples:5d}')
    logger.info(f'Number of samples: {n_samples:5d}')
    print(f'Accuracy: {tot_acc / n_samples:.4f}')
    logger.info(f'Accuracy: {tot_acc / n_samples:.4f}')
    print(f'Msg Accuracy: {tot_msg_acc / n_samples:.4f}')
    logger.info(f'Msg Accuracy: {tot_msg_acc / n_samples:.4f}')

    print(f'Detection F1: {tot_detect_f1 / n_samples:.4f}')
    logger.info(f'Detection F1: {tot_detect_f1 / n_samples:.4f}')
    print(f'Detection TPR: {tot_detect_tpr / n_samples:.4f}')
    logger.info(f'Detection TPR: {tot_detect_tpr / n_samples:.4f}')

    print(f'ONION Adv Accuracy: {adv_acc / n_samples:.4f}')
    logger.info(f'ONION Adv Accuracy: {adv_acc / n_samples:.4f}')
    print(f'ONION Adv Msg Accuracy: {adv_msg_acc / n_samples:.4f}')
    logger.info(f'ONION Adv Msg Accuracy: {adv_msg_acc / n_samples:.4f}')

    for key in adv_codebleu_res.keys():
        logger.info('Transformed vs Adv')
        logger.info(f'{key}: {adv_codebleu_res[key] / n_samples:.4f}')
        logger.info('-' * 80)

    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    assert len(test_instances) == len(adv_test_objs)
    fo_name = f'./results/{ckpt_name}_onion.jsonl'
    with open(fo_name, 'w') as fo:
        for instance in adv_test_objs:
            fo.write(json.dumps(instance, ensure_ascii=False) + '\n')
    print(f'Wrote adversarial examples to {fo_name}')
