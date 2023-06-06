import copy
import os
from lxml import etree
from lxml.etree import Element

flag = True
doc = None
ns = {'src': 'http://www.srcML.org/srcML/src'}


def init_parse(e):
    global doc
    doc = etree.parse(e)
    e = etree.XPathEvaluator(doc, namespaces={'src': 'http://www.srcML.org/srcML/src'})
    return e


def get_expr(e):
    return e('//src:expr')


def trans_tree(e, ignore_list=[], instances=None):
    global flag
    flag = False
    expr_elems = [
        get_expr(e) if instances is None else (instance[0] for instance in instances)
    ]
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []

    for item in expr_elems:
        for expr_elem in item:
            #
            expr_elem_prev = expr_elem.getprevious()
            expr_elem_prev = expr_elem_prev if expr_elem_prev is not None else expr_elem
            expr_elem_prev_path = tree_root.getpath(expr_elem_prev)
            if expr_elem_prev_path in ignore_list:
                continue

            # expr loop，find name,operator,operator,name
            # temp = ++i;
            if len(expr_elem) >= 4:
                tag = etree.QName(expr_elem.getparent())
                if tag.localname == 'condition':
                    continue
                var = []
                a_flag = True
                for elem in expr_elem[:-1]:
                    if elem.text == '+' or elem.text == '-' or elem.text == '*' or elem.text == '/':
                        a_flag = False
                        break
                if a_flag == False:
                    continue
                for elem in expr_elem:
                    if elem.text == '=':
                        index = elem.getparent().index(elem)
                        parent = elem.getparent()
                        if parent[index + 1].text == '++' or parent[index +
                                                                    1].text == '--':
                            block = parent.getparent().getparent().getparent()
                            if not block.text:
                                block.text = '{'
                                block_content = parent.getparent().getparent()
                                block_content.tail = '}'
                            for var_elem in expr_elem[index + 2:]:
                                var.append(copy.deepcopy(var_elem))
                            var[-1].tail = ';'
                            parent.insert(0, parent[index + 1])
                            node = Element("call")
                            for var_e in var:
                                node.append(var_e)
                            parent.insert(1, node)
                            flag = True
                            #
                            new_ignore_list.append(expr_elem_prev_path)
            # temp = i++;
            if len(expr_elem) >= 4:
                tag = etree.QName(expr_elem.getparent())
                if tag.localname == 'condition':
                    continue
                var = []
                for elem in expr_elem:
                    if elem.text == '=':
                        index = elem.getparent().index(elem)
                        if expr_elem[-1].text == '++' or expr_elem[-1].text == '--':
                            block = parent.getparent().getparent().getparent()
                            if not block.text:
                                block.text = '{'
                                block_content = parent.getparent().getparent()
                                block_content.tail = '}'
                            for var_elem in expr_elem[index + 1:-1]:
                                var.append(copy.deepcopy(var_elem))
                            var[-1].tail = ';'
                            node = Element("call")
                            for var_e in var:
                                node.append(var_e)
                            expr_elem.insert(index + 1, node)
                            flag = True
                            #
                            new_ignore_list.append(expr_elem_prev_path)
    return flag, tree_root, new_ignore_list


def save_tree_to_file(tree, path):
    with open(path, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


def count(e):
    count_num = 0
    # get all expr
    expr_elems = get_expr(e)
    for expr_elem in expr_elems:
        # expr loop，find name,operator,operator,name
        # temp = ++i;
        if len(expr_elem) >= 4:
            tag = etree.QName(expr_elem.getparent())
            if tag.localname == 'condition':
                continue
            a_flag = True
            for elem in expr_elem[:-1]:
                if elem.text == '+' or elem.text == '-' or elem.text == '*' or elem.text == '/':
                    a_flag = False
                    break
            if a_flag == False:
                continue
            for elem in expr_elem:
                if elem.text == '=':
                    index = elem.getparent().index(elem)
                    parent = elem.getparent()
                    if parent[index + 1].text == '++' or parent[index + 1].text == '--':
                        count_num += 1
        if len(expr_elem) >= 4:
            tag = etree.QName(expr_elem.getparent())
            if tag.localname == 'condition':
                continue
            for elem in expr_elem:
                if elem.text == '=':
                    if expr_elem[-1].text == '++' or expr_elem[-1].text == '--':
                        count_num += 1
    return count_num


def get_number(xml_path):
    xmlfilepath = os.path.abspath(xml_path)
    e = init_parse(xmlfilepath)
    return count(e)


def program_transform(input_xml_path: str, output_xml_path: str = './style/style.xml'):
    e = init_parse(input_xml_path)
    trans_tree(e)
    save_tree_to_file(doc, output_xml_path)