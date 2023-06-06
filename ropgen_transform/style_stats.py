from collections import defaultdict
from typing import List, Dict, Optional


def calculate_proportion(file_styles: List[Dict[int, Dict[str, int]]],
                         num_files: Optional[int] = None):
    """normalizes a list of file styles to a single dict of style attribute proportions.

    Args:
        file_styles: A list of dict. Each dict is a mapping from style attribute id
        to a dict containing the occurrences of each style attribute value.
        num_files (optional): If is None, will be determined by length of file_styles.
        Defaults to None.
    """
    if num_files is None:
        num_files = len(file_styles)

    style_counter = defaultdict(lambda: defaultdict(int))
    # counting occurrences of each style attribute
    for file_idx in range(len(file_styles)):
        cur_style = file_styles[file_idx]
        for style_key in cur_style:
            if style_key not in cur_style:
                continue

            # special handling for numeric attribute #23 (num of lines per function)
            if style_key == 23:
                for j in range(0, 2):
                    style_counter[style_key]['23'][j] += cur_style[style_key]['23'][j]
                continue

            # accumulate the number of occurrences of each style attribute
            for key in cur_style[style_key]:
                style_counter[style_key][key] += cur_style[style_key][key]

    for style_key in style_counter:
        # another special handling for #23
        if style_key == 23:
            for j in range(0, 2):
                avg_val = (style_counter[style_key]['23'][j] / num_files)
                style_counter[style_key]['23'][j] = round(avg_val, 1)
            continue

        # special handling for #4, #11 and #12
        if style_key == 4:
            if style_counter[style_key]['4.1'] > 0:
                style_counter[style_key]['4.1'] = 100.0
                continue
        if style_key == 11:
            if style_counter[style_key]['11.1'] > 0:
                style_counter[style_key]['11.1'] = 100.0
                continue
        if style_key == 12:
            if style_counter[style_key]['12.1'] > 0:
                style_counter[style_key]['12.1'] = 100.0
                continue

        total = sum(style_counter[style_key].values())
        if total == 0:
            continue
        for key in style_counter[style_key]:
            style_counter[style_key][key] = round(
                (style_counter[style_key][key] / total) * 100, 1)
    return style_counter


def get_dominating_styles(style_proportions: Dict[int, Dict]) -> List[str]:
    dominating_styles = []

    for style_key in style_proportions:
        style_dict = style_proportions[style_key]

        for style_attr in style_dict:
            num_attrs = len(style_dict)
            if style_attr == '21.1' or style_attr == '21.2':
                if style_dict[style_attr] > 0:
                    dominating_styles.append(style_attr)
            elif style_attr == '23':
                dominating_styles.append({'23': style_dict[style_attr]})
            elif num_attrs == 2:
                if style_dict[style_attr] >= 70:
                    dominating_styles.append(style_attr)
            elif num_attrs == 4:
                if style_dict[style_attr] >= 50:
                    dominating_styles.append(style_attr)
            elif num_attrs == 5:
                if style_dict[style_attr] >= 50:
                    dominating_styles.append(style_attr)

    return dominating_styles
