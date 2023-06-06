import sys
import inflection
from lxml import etree
from ropgen_transform.xml_utils import load_doc, init_parser, XML_NS

KEYWORDS = {
    "auto", "abstract", "assert", "bool", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extern", "extends", "final", "finally", "float", "for", "goto", "if", "elif",
    "implements", "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try",
    "void", "volatile", "while", "ifndef", "ifdef", "endif", "define", "undef", "include",
    "pragma", "typedef", "unsigned", "using"
}


def get_decl_stmts(e):
    return e('//src:decl_stmt')


def get_names(e):
    return e('//src:name')


def get_decl(elem):
    return elem.xpath('src:decl', namespaces=XML_NS)


def get_declname(elem):
    return elem.xpath('src:name', namespaces=XML_NS)


def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


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
    return '_' in name.strip('_')


def is_init_underscore(name):
    return name[0] == '_' and name[1:].strip('_') != ''


def is_init_dollar(name):
    return name[0] == '$' and name[1:].strip('$') != ''


def underscore_to_initcap(name):
    if not is_underscore(name):
        return name
    new_name = ''.join(name[0].upper())
    is_prev_underscore = False
    for ch in name[1:]:
        if ch == '_':
            is_prev_underscore = True
        else:
            if is_prev_underscore:
                new_name += ch.upper()
                is_prev_underscore = False
            else:
                new_name += ch
    return new_name


def underscore_to_camel(name):
    if not is_underscore(name):
        return name
    new_name = ''
    is_prev_underscore = False
    for ch in name:
        if ch == '_':
            is_prev_underscore = True
        else:
            if is_prev_underscore:
                new_name += ch.upper()
                is_prev_underscore = False
            else:
                new_name += ch
    return new_name


def underscore_to_init_symbol(name, symbol):
    if not is_underscore(name):
        return name
    return symbol + name


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
    new_name = ''
    for ch in name:
        if ch.isupper():
            new_name += '_' + ch.lower()
        else:
            new_name += ch
    return new_name


def initcap_to_underscore(name):
    if not is_initcap(name):
        return name
    new_name = ''.join(name[0].lower())
    for ch in name[1:]:
        if ch.isupper():
            new_name += '_' + ch.lower()
        else:
            new_name += ch
    return new_name


def to_upper(name):
    return name.upper()


def get_decls(e):
    decls = []
    decl_stmts = get_decl_stmts(e)
    for decl_stmt in decl_stmts:
        decl_list = get_decl(decl_stmt)
        for decl in decl_list:
            decls.append(decl)
    return decls


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
            raise RuntimeError(f'cannot convert variable {name} to snake_case')
        else:
            return name


def normalized_to_init_cap(name: str):
    if is_all_lowercase(name):
        return name[0].upper() + name[1:]
    elif is_all_uppercase(name):
        return name[0] + name[1:].lower()
    elif is_underscore(name):
        return underscore_to_initcap(name)
    else:
        raise RuntimeError(f'{name} is not a normalized name')


def normalized_to_init_symbol(name: str, symbol: str):
    if is_all_lowercase(name):
        return symbol + name
    elif is_all_uppercase(name):
        return symbol + name
    elif is_underscore(name):
        return underscore_to_init_symbol(name, symbol)
    else:
        raise RuntimeError(f'{name} is not a normalized name')


def transform_all(evaluator, dst_style):
    decls = [get_decls(evaluator)]
    tree_root = evaluator('/*')[0].getroottree()

    for item in decls:
        for decl in item:
            if len(get_declname(decl)) == 0:
                continue
            name_node = get_declname(decl)[0]
            name_text = name_node.text
            if name_text is None:
                name_node = get_declname(name_node)[0]
                name_text = name_node.text

            normalized_name_text = normalize_name(name_text, allow_raising=True)
            new_name = normalized_name_text
            if dst_style == '1.1':
                new_name = underscore_to_camel(normalized_name_text)
            elif dst_style == '1.2':
                new_name = normalized_to_init_cap(normalized_name_text)
            elif dst_style == '1.3':
                new_name = normalized_name_text
            elif dst_style == '1.4':
                new_name = normalized_to_init_symbol(normalized_name_text, '_')
            elif dst_style == '1.5':
                new_name = normalized_to_init_symbol(normalized_name_text, '$')

            if new_name in KEYWORDS:
                new_name = new_name + '_'

            whitelist = ['main', 'size', 'operator', 'case']
            names = get_names(evaluator)
            name_list = [name.text for name in names]
            if new_name in whitelist or new_name in name_list:
                continue
            name_node.text = new_name

            for name in names:
                if name.text == name_text:
                    name.text = new_name

    return tree_root


# entry and dispatcher function
# arguments 'ignore_list' and 'instances' are pretty much legacy code and can be ignored
# argument 'e' is obtained by calling init_parser(srcML XML path)
# 'src_style' the style of source author
# 'dst_style' the style of target author
def transform(evaluator, src_style, dst_style):
    decls = [get_decls(evaluator)]
    tree_root = evaluator('/*')[0].getroottree()

    for item in decls:
        for decl in item:
            if len(get_declname(decl)) == 0:
                continue
            name_node = get_declname(decl)[0]
            name_text = name_node.text
            if name_text is None:
                name_node = get_declname(name_node)[0]
                name_text = name_node.text

            src_dst_tuple = (src_style, dst_style)
            if src_dst_tuple == ('1.1', '1.2'):
                new_name = underscore_to_initcap(camel_to_underscore(name_text))
            elif src_dst_tuple == ('1.1', '1.3'):
                new_name = camel_to_underscore(name_text)
            elif src_dst_tuple == ('1.1', '1.4'):
                new_name = underscore_to_init_symbol(camel_to_underscore(name_text), '_')
            elif src_dst_tuple == ('1.1', '1.5'):
                new_name = underscore_to_init_symbol(camel_to_underscore(name_text), '$')
            elif src_dst_tuple == ('1.2', '1.1'):
                new_name = underscore_to_camel(initcap_to_underscore(name_text))
            elif src_dst_tuple == ('1.2', '1.3'):
                new_name = initcap_to_underscore(name_text)
            elif src_dst_tuple == ('1.2', '1.4'):
                new_name = underscore_to_init_symbol(initcap_to_underscore(name_text),
                                                     '_')
            elif src_dst_tuple == ('1.2', '1.5'):
                new_name = underscore_to_init_symbol(initcap_to_underscore(name_text),
                                                     '$')
            elif src_dst_tuple == ('1.3', '1.1'):
                new_name = underscore_to_camel(name_text)
            elif src_dst_tuple == ('1.3', '1.2'):
                new_name = underscore_to_initcap(name_text)
            elif src_dst_tuple == ('1.3', '1.4'):
                new_name = underscore_to_init_symbol(name_text, '_')
            elif src_dst_tuple == ('1.3', '1.4'):
                new_name = underscore_to_init_symbol(name_text, '$')
            elif src_dst_tuple == ('1.4', '1.1'):
                new_name = underscore_to_camel(init_symbol_to_underscore(name_text))
            elif src_dst_tuple == ('1.4', '1.2'):
                new_name = underscore_to_initcap(init_symbol_to_underscore(name_text))
            elif src_dst_tuple == ('1.4', '1.3'):
                new_name = init_symbol_to_underscore(name_text)
            elif src_dst_tuple == ('1.4', '1.5'):
                new_name = underscore_to_init_symbol(init_symbol_to_underscore(name_text),
                                                     '$')
            elif src_dst_tuple == ('1.5', '1.1'):
                new_name = underscore_to_camel(init_symbol_to_underscore(name_text))
            elif src_dst_tuple == ('1.5', '1.2'):
                new_name = underscore_to_initcap(init_symbol_to_underscore(name_text))
            elif src_dst_tuple == ('1.5', '1.3'):
                new_name = init_symbol_to_underscore(name_text)
            elif src_dst_tuple == ('1.5', '1.4'):
                new_name = underscore_to_init_symbol(init_symbol_to_underscore(name_text),
                                                     '_')

            whitelist = ['main', 'size', 'operator', 'case']
            names = get_names(evaluator)
            name_list = [name.text for name in names]
            if new_name in whitelist or new_name in name_list:
                continue
            name_node.text = new_name

            for name in names:
                if name.text == name_text:
                    name.text = new_name

    return tree_root


def transform_standalone_stmts(e):
    for decl in get_decls(e):
        name_node = get_declname(decl)[0]
        name_text = name_node.text
        if name_text is None:
            name_node = get_declname(name_node)[0]
            name_text = name_node.text
        print(initcap_to_underscore(name_text))


def program_transform(program_path: str,
                      src_style: str,
                      dst_style: str,
                      out_fname: str = './style/style.xml'):
    doc = load_doc(program_path)
    evaluator = init_parser(doc)
    transform(evaluator, src_style, dst_style)
    save_tree_to_file(doc, out_fname)


def program_transform_all(program_path: str,
                          dst_style: str,
                          out_fname: str = './style/style.xml'):
    doc = load_doc(program_path)
    evaluator = init_parser(doc)
    transform_all(evaluator, dst_style)
    save_tree_to_file(doc, out_fname)


def etree_transform(evaluator, dst_style: str):
    transform_all(evaluator, dst_style)
    return evaluator


if __name__ == '__main__':
    doc = load_doc(sys.argv[1])
    e = init_parser(doc)
    transform(e, '1.3', '1.1')
    save_tree_to_file(doc, './var_name_style.xml')
