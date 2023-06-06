import sys
import os
from lxml import etree

ns = {
    'src': 'http://www.srcML.org/srcML/src',
    'cpp': 'http://www.srcML.org/srcML/cpp',
    'pos': 'http://www.srcML.org/srcML/position'
}
doc = None
flag = False


def init_parser(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc)
    for k, v in ns.items():
        e.register_namespace(k, v)
    return e


def get_decl_stmts(e):
    return e('//src:decl_stmt')


def get_decl(elem):
    return elem.xpath('src:decl', namespaces=ns)


def get_typename(elem):
    return elem.xpath('src:type', namespaces=ns)


def get_typespec(elem):
    return elem.xpath('src:type/src:specifier', namespaces=ns)


def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


def get_multi_decl_stmts(e):
    multi_decl_stmts = []
    decl_stmts = get_decl_stmts(e)
    for decl_stmt in decl_stmts:
        decls = get_decl(decl_stmt)
        if len(decls) > 1:
            multi_decl_stmts.append(decl_stmt)
    return multi_decl_stmts


# transform multiple declarations in one statement into multiple statements
#int i, j;
#int i; int j;
# Step 1: get all statements containing multiple declarations
# Step 2: for each one of these statements, get its type name
# Step 3: iterate over the children of the statement, and if a comma is encountered
#         change it to semicolon + type name + space


def transform_standalone_stmts(e, ignore_list=[], instances=None):
    global flag
    flag = False
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    multi_decl_stmts = [
        get_multi_decl_stmts(e) if instances is None else
        (instance[0] for instance in instances)
    ]
    for item in multi_decl_stmts:
        for decl_stmt in item:
            decl_stmt_prev = decl_stmt.getprevious()
            decl_stmt_prev = decl_stmt_prev if decl_stmt_prev is not None else decl_stmt
            decl_stmt_prev_path = tree_root.getpath(decl_stmt_prev)
            if decl_stmt_prev_path in ignore_list: continue
            flag = True
            decl = get_decl(decl_stmt)[0]
            if len(get_typename(decl)) == 0: continue
            type_node = get_typename(decl)[0]
            type_spec_node = get_typespec(decl)
            type_itertext = type_node.itertext()
            type_text = ''.join([item for item in type_itertext if item != '*'])
            # type_spec = type_spec_node[0].text if len(type_spec_node) > 0 else ''
            # print(type_text)
            prev_node = None
            last_type_node = None
            for child in decl_stmt.xpath('child::node()'):
                if not isinstance(child, etree._ElementUnicodeResult):
                    prev_node = child
                    last_type_node = child
                    continue
                if child.strip() == ',':
                    if type_text is None: continue
                    prev_node.tail = '; ' + type_text + ' '
                prev_node = child
            new_ignore_list.append(decl_stmt_prev_path)
            #print(etree.tostring(decl_stmt).decode('utf8'))
    return flag, tree_root, new_ignore_list


def xml_file_path(xml_path):
    global flag
    save_xml_file = './transform_xml_file/var_init_split'
    transform_java_file = './target_author_file/transform_java/var_init_split'
    if not os.path.exists(transform_java_file):
        os.mkdir(transform_java_file)
    if not os.path.exists(save_xml_file):
        os.mkdir(save_xml_file)
    for xml_path_elem in xml_path:
        xmlfilepath = os.path.abspath(xml_path_elem)
        e = init_parser(xmlfilepath)
        flag = False
        transform_standalone_stmts(e)
        if flag:
            str = xml_path_elem.split('/')[-1]
            sub_dir = xml_path_elem.split('/')[-2]
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_tree_to_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file


def program_transform(input_xml_path: str, output_xml_path: str = './style/style.xml'):
    e = init_parser(input_xml_path)
    transform_standalone_stmts(e)
    save_tree_to_file(doc, output_xml_path)


if __name__ == '__main__':
    e = init_parser(sys.argv[1])
    transform_standalone_stmts(e)
    save_tree_to_file(doc, 'var_init_split.xml')
