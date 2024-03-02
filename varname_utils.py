import inflection


def init_symbol_to_underscore(name):
    if not is_init_underscore(name) and not is_init_dollar(name):
        return name
    init_char = name[0]
    new_name = name.strip(init_char)

    if is_camel_case(new_name):
        return camel_to_underscore(new_name)
    elif is_initcap(new_name):
        return initcap_to_underscore(new_name)
    return new_name


def camel_to_underscore(name):
    if not is_camel_case(name):
        return name
    new_name = ""
    for ch in name:
        if ch.isupper():
            new_name += "_" + ch.lower()
        else:
            new_name += ch
    return new_name


def initcap_to_underscore(name):
    if not is_initcap(name):
        return name
    new_name = "".join(name[0].lower())
    for ch in name[1:]:
        if ch.isupper():
            new_name += "_" + ch.lower()
        else:
            new_name += ch
    return new_name


def is_all_lowercase(name):
    return name.lower() == name


def is_all_uppercase(name):
    return name.upper() == name


def is_camel_case(name):
    if is_all_lowercase(name):
        return False
    if not name[0].isalpha():
        return False
    return inflection.camelize(name, uppercase_first_letter=False) == name


def is_initcap(name):
    if is_all_uppercase(name):
        return False
    if not name[0].isalpha():
        return False
    return inflection.camelize(name, uppercase_first_letter=True) == name


def is_underscore(name):
    return "_" in name.strip("_")


def is_init_underscore(name):
    return name[0] == "_" and name[1:].strip("_") != ""


def is_init_dollar(name):
    return name[0] == "$" and name[1:].strip("$") != ""


def normalize_name(name: str, allow_raising: bool = False):
    if is_camel_case(name):
        return camel_to_underscore(name)
    elif is_initcap(name):
        return initcap_to_underscore(name)
    elif is_init_underscore(name) or is_init_dollar(name):
        return init_symbol_to_underscore(name)
    elif is_underscore(name):
        return name
    elif is_all_lowercase(name):
        return name
    elif is_all_uppercase(name):
        return name
    else:
        if allow_raising:
            raise RuntimeError(f"cannot convert variable {name} to snake_case")
        else:
            return name
