import sys
import json
from code_tokenizer import CodeTokenizer, tokens_to_strings
from tqdm import tqdm
from collections import defaultdict
from metrics import calc_code_bleu


def main(args):
    if len(args) != 1:
        print('Usage: python jsonl_codebleu_eval.py [jsonl_file]')
        return
    code_tokenizer = CodeTokenizer(lang='java')

    codebleu_res = defaultdict(int)
    jsonl_file = args[0]
    with open(jsonl_file, 'r', encoding='utf-8') as fi:
        objs = [json.loads(line) for line in fi.readlines()]

    tot_samples = 0
    for obj in tqdm(objs):
        original = obj['original_string']
        tokens = code_tokenizer.get_tokens(original)[0]
        original = tokens_to_strings(tokens)

        after_watermark = obj['after_watermark']

        res = calc_code_bleu.evaluate_per_example(reference=original,
                                                  hypothesis=after_watermark,
                                                  lang='java')

        for key, val in res.items():
            codebleu_res[key] += val
        tot_samples += 1

    assert tot_samples == len(objs)
    for key, val in codebleu_res.items():
        codebleu_res[key] = val / tot_samples

    for key, val in codebleu_res.items():
        print(f'{key}: {val:.4f}')


if __name__ == '__main__':
    main(sys.argv[1:])
