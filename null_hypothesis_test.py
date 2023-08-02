import sys
import torch
import pickle
from scipy.stats import binomtest
from typing import Dict, List

SEP_1 = 3 * 4
SEP_2 = 8 * 4
SEP_3 = 33 * 4


def null_hypothesis_test(fail_chance: float, pickle_path: str):
    repowise_long_gt: Dict[str, List[int]]
    repowise_long_msg: Dict[str, List[int]]
    pvalues_1 = dict()
    pvalues_2 = dict()
    pvalues_3 = dict()
    pvalues_4 = dict()

    repowise_long_msg, repowise_long_gt = pickle.load(open(pickle_path, 'rb'))

    n_funcs = []
    total_matched_bits = 0
    total_bits = 0
    for repo, msg in repowise_long_msg.items():
        # print(repo)
        # print(msg, repowise_long_gt[repo])

        total_msg_len = len(msg)
        n_funcs.append(total_msg_len // 4)
        matched_msg_len = 0
        for i in range(total_msg_len):
            msg_bit = msg[i]
            gt_bit = repowise_long_gt[repo][i]

            if torch.rand(1) < fail_chance:
                # simulated effect of an attack that alters some bits
                msg_bit = 1 - msg_bit

            if msg_bit == gt_bit:
                matched_msg_len += 1

        total_matched_bits += matched_msg_len
        total_bits += total_msg_len

        if total_msg_len <= SEP_1:
            pvalues_1[repo] = binomtest(matched_msg_len, total_msg_len, 0.5).pvalue
        elif total_msg_len <= SEP_2:
            pvalues_2[repo] = binomtest(matched_msg_len, total_msg_len, 0.5).pvalue
        elif total_msg_len <= SEP_3:
            pvalues_3[repo] = binomtest(matched_msg_len, total_msg_len, 0.5).pvalue
        else:
            pvalues_4[repo] = binomtest(matched_msg_len, total_msg_len, 0.5).pvalue

    # print(n_funcs)

    thresholds = [0.01]
    print(f'Fail chance: {fail_chance:.2f}')
    print(f'Bitwise Accuracy: {total_matched_bits/total_bits*100:.2f}%')
    for thres in thresholds:

        all_passed = 0
        all_total = 0
        labels = ['1-3', '4-8', '9-33', '34+']
        for i, pvalues in enumerate([pvalues_1, pvalues_2, pvalues_3, pvalues_4]):
            passed = 0
            total = 0
            label = labels[i]
            for repo, pvalue in pvalues.items():
                if pvalue < thres:
                    passed += 1
                    all_passed += 1
                total += 1
                all_total += 1
            print(f'{label}: Threshold {thres:.3f}: {passed}/{total}'
                  f' ({passed/total*100:.2f}%)')

        print(f'Macro: Threshold {thres:.3f}: {all_passed}/{all_total}'
              f' ({all_passed/all_total*100:.2f}%)')
        print()


if __name__ == '__main__':
    torch.manual_seed(42)

    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <pickle_path>')
        exit(0)

    null_hypothesis_test(fail_chance=0, pickle_path=sys.argv[1])
