"""
if(a && b) ->  if(a){if(b){...}}

Splitting if conditional statements

"""

import os
from lxml import etree
from lxml.etree import Element

ns = {"src": "http://www.srcML.org/srcML/src"}
doc = None
flag = False  # Tag, indicating that for can be converted


# # parsing XML file into tree
def init_parse(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc, namespaces={'src': 'http://www.srcML.org/srcML/src'})
    return e


def get_if(e):
    return e('//src:if')


# get if's condition expression
def get_expr(elem):
    return elem.xpath('src:condition/src:expr', namespaces=ns)


# get if's executable statement
def get_block_content(elem):
    return elem.xpath('src:block', namespaces=ns)


# get if's executable statement content
def get_stmt(elem):
    return elem.xpath('src:block_content', namespaces=ns)


def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


def trans_1(elem, expr, if_elem):
    # get if's executable statement
    if len(get_block_content(if_elem)) > 0:
        block = get_block_content(if_elem)[0]
        condition_index = expr.index(elem)
        # delete '&&'
        expr.remove(elem)
        # get the second condition
        second_expr = expr[condition_index:]
        # add ')' at the end of the condition
        second_expr[-1].tail = ')'
        if len(get_stmt(block)) != 0:
            stmt = get_stmt(block)[0]
        else:
            stmt = Element('block_content')
            stmt.tail = '}'
            stmt.append(block[0])
        if len(stmt) > 0:
            stmt[-1].tail = '}'
        # add the if statement to the first if executable statement
        node = Element('if')
        node.text = 'if'
        block.insert(0, node)
        # add the second condition to the second if statement
        node = Element('condition')
        node.text = '('
        block[0].append(node)
        for elem in second_expr:
            block[0][0].append(elem)
        # add executable statement at the second if
        node = Element('block')
        node.text = '{'
        block[0].append(node)
        # add the first executable statement at the second executable statement
        block[0][1].append(stmt)


# start to transform
def trans_tree(e, ignore_list=[], instances=None):
    global flag
    flag = False
    # get the root of the tree, then take the path to use
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    # get all if statement
    if_elems = [
        get_if(e) if instances is None else (instance[0] for instance in instances)
    ]
    for item in if_elems:
        for if_elem in item:

            if_elem_prev = if_elem.getprevious()
            if_elem_prev = if_elem_prev if if_elem_prev is not None else if_elem
            if_elem_prev_path = tree_root.getpath(if_elem_prev)
            if if_elem_prev_path in ignore_list:
                continue

            if_stmt = if_elem.getparent()
            if len(if_stmt) == 1:
                # get if's condition, judge whether to split
                if len(get_expr(if_elem)) == 0: continue
                expr = get_expr(if_elem)[0]
                # judge whether there are '&&' in the condition
                flag_elem = True
                for elem in expr:
                    if elem.text == '||' or elem.text == '|':
                        flag_elem = False
                        break
                if flag_elem is True:
                    for elem in expr:
                        if elem.text == '&&':
                            new_ignore_list.append(if_elem_prev_path)
                            flag = True
                            trans_1(elem, expr, if_elem)
                            # just split the first '&&'
                            break
    return flag, tree_root, new_ignore_list


def count(e):
    count_num = 0
    if_elems = get_if(e)
    for if_elem in if_elems:
        if_stmt = if_elem.getparent()
        if len(if_stmt) == 1:
            if len(get_expr(if_elem)) == 0: continue
            expr = get_expr(if_elem)[0]
            flag_elem = True
            for elem in expr:
                if elem.text == '||' or elem.text == '|':
                    flag_elem = False
                    break
            if flag_elem == True:
                for elem in expr:
                    if elem.text == '&&':
                        count_num += 1
    return count_num


# calculate the number of if
def get_number(xml_path):
    xmlfilepath = os.path.abspath(xml_path)
    e = init_parse(xmlfilepath)
    return count(e)


def etree_transform(evaluator, dst_style: str):
    trans_tree(evaluator)
    return evaluator


# the program's input port
def program_transform(program_path, output_xml_path: str = './style/style.xml'):
    e = init_parse(program_path)
    trans_tree(e)
    save_tree_to_file(doc, output_xml_path)
