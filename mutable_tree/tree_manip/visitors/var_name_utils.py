import re
import inflection
from .keywords import ALL_KEYWORDS_SET


def is_underscore_case(name) -> bool:
    return name[0] == "_" and name[1:].strip("_") != ""


def remove_preceding_underscores(name: str) -> str:
    idx = 0
    while idx < len(name) and name[idx] == "_":
        idx += 1
    name = name[idx:]
    if name == "":
        return "_"
    else:
        return name


def sanitize_name_for_styling(name):
    # sometimes raw keywords in code cause tree-sitter to explode
    # so we further sanitize the name by capitalizing keywords
    name = sanitize_name(name)
    if name in ALL_KEYWORDS_SET:
        if name[0].islower():
            name = name[0].upper() + name[1:]
        else:
            name = name + "_"
    return name


def sanitize_name(name):
    # https://github.com/eliphatfs/torch.redstone
    return re.sub(r"\W|^(?=\d)", "_", name)


def normalize_name(name: str) -> str:
    if is_underscore_case(name):
        new_name = name[1:]
    else:
        new_name = inflection.underscore(name)
    return sanitize_name(new_name)
