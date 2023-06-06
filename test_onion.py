import os
import math
import torch
import argparse
import transformers
import numpy as np

from tqdm import tqdm

KW_LANG = 'java'
KW_PATH = f'./metrics/keywords/{KW_LANG}.txt'
with open(KW_PATH, 'r') as f:
    KEYWORDS = set([kw.strip() for kw in f.readlines()])


class GPT2LM:

    def __init__(self, device=None, lm_model: str = 'microsoft/CodeGPT-small-java'):
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(lm_model)
        self.lm = transformers.GPT2LMHeadModel.from_pretrained(lm_model)
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


def get_PPL(instances, gpt_lm):
    with torch.no_grad():
        all_PPL = []
        for i, instance in enumerate(tqdm(instances)):
            sent, update = instance
            split_sent = sent.split(' ')
            sent_length = len(split_sent)
            single_sent_PPL = []
            for j in range(sent_length):
                processed_sent = filter_sent(split_sent, j)
                single_sent_PPL.append(gpt_lm(processed_sent))
            all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(instances)
    return all_PPL


def get_processed_sent(flag_li, orig_sent, update):
    tp = 0
    fp = 0
    fn = 0
    hit = False
    suspects = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 0:
            # poisoned word
            if word.isidentifier() and word not in KEYWORDS:
                suspects.append(word)
            if word == update[1] and not hit:
                tp += 1
                hit = True
            elif word != update[1]:
                fp += 1
    if not hit:
        fn += 1
    print(suspects)
    print(f'[{update[1]}]')

    return tp, fp, fn


def get_processed_poison_data(all_PPL, instances, thres):
    tot_f1 = 0
    tot_tpr = 0
    n_samples = 0
    for i, PPL_li in enumerate(all_PPL):
        orig_sent, update = instances[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        
        print()
        print(whole_sentence_PPL)
        
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= thres:
                flag_li.append(0)  # 0: is poisoned
            else:
                flag_li.append(1)  # 1: natural

        assert len(flag_li) == len(orig_split_sent)
        tp, fp, fn = get_processed_sent(flag_li, orig_split_sent, update)

        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0
        tpr = tp / (tp + fn) if tp + fn > 0 else 0
        tot_f1 += f1
        tot_tpr += tpr

        n_samples += 1

    return tot_f1 / n_samples, tot_tpr / n_samples


if __name__ == '__main__':
    # logs/undergrad/2023-04-29-01-20-08-eval-4bit_func_42_github_java_funcs.log
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_file', default='record.log')
    args = parser.parse_args()

    # LM = 'gpt2'
    LM = 'microsoft/CodeGPT-small-java'

    # gpt_lm = GPT2LM(device='cuda', lm_model=LM)
    file_path = args.record_file
    instances = []
    with open(args.record_file, 'r') as fi:
        lines = fi.readlines()
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if line.startswith('Transformed Code:'):
            transformed = line.replace('Transformed Code: ', '').strip()
            cursor += 1
            line = lines[cursor]
            assert line.startswith('Updates:')
            update = line.replace('Updates: ', '').strip()
            update = eval(update)
            instances.append((transformed, update))
        cursor += 1

    # all_PPL = get_PPL(instances, gpt_lm)
    # torch.save(all_PPL, 'all_PPL_codegpt.pt')
    all_PPL = torch.load('all_PPL_text.pt')

    print('thres,f1,tpr')
    # for thres in list(range(-5, 0)) + [-0.5, -0.1, -0.01]:
    for thres in list(range(-10, 0)) + [-0.5]:
        f1, tpr = get_processed_poison_data(all_PPL, instances, thres)
        print(f'{thres},{f1},{tpr}')
