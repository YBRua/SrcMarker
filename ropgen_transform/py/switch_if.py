"""
    switch -> if 

"""

import copy
import os

from lxml import etree
from lxml.etree import Element

ns = {'src': "http://www.srcML.org/srcML/src"}
doc = None
flag_switch = True  # judge whether switch can be transform
flag = True  # judge whether the transformation is successful


def init_parse(path):
    global doc
    doc = etree.parse(path)
    e = etree.XPathEvaluator(doc, namespaces={'src': 'http://www.srcML.org/srcML/src'})
    return e


def get_switch(e):
    return e("//src:switch")


def get_variable(elem):
    return elem.xpath('src:condition', namespaces=ns)


def get_case(elem):
    return elem.xpath('src:block/src:block_content/src:case', namespaces=ns)


def get_block_content(elem):
    return elem.xpath("src:block/src:block_content", namespaces=ns)


def get_default(elem):
    return elem.xpath("src:block/src:block_content/src:default", namespaces=ns)


# default structure transform
def transform_default(default_elem):
    if len(default_elem) > 0:
        default_elem[0].tag = 'else'
        default_elem[0].text = 'else'
        index = default_elem[0].getparent().index(default_elem[0])
        node = default_elem[0].getparent()[index + 1]
        tag = etree.QName(node)
        if tag.localname != 'block':
            node_elem = Element('block')
            node_elem.text = '{'
            for elem in default_elem[0].getparent()[index:]:
                tag = etree.QName(elem)
                if tag.localname == 'expr_stmt':
                    node_elem.append(elem)
                if elem.text == 'break;':
                    elem_index = elem.getparent().index(elem)
                    del elem.getparent()[elem_index]
                    break
            node_elem.tail = '\n}'
            default_elem[0].getparent().insert(index + 1, node_elem)


# case structure transform
def transform_case(case_elems, var):
    # transform the first case
    case_first = case_elems[0]
    case_first.tag = 'if'
    case_first.text = 'if'
    transform_case_content(case_first, var)
    # transform the case after the first one
    for elem in case_elems[1:]:
        elem.tag = 'if'
        elem.text = 'else if'
        transform_case_content(elem, var)


# case content transform
def transform_case_content(case_elem, var):
    global flag
    # deep copy case's condition variable
    var_elem = copy.deepcopy(var)
    operate_elem = Element('operate')
    operate_elem.text = '=='
    var_elem.append(operate_elem)
    # the value of case is added to the condition
    for case_child in case_elem:
        var_elem.append(case_child)
    # delete “：”
    var_elem[-1].tail = ')'
    # put the condition in the if condition sentence
    case_elem.append(var_elem)
    # modify case's executable sentence
    index = case_elem.getparent().index(case_elem)
    node = case_elem.getparent()[index + 1]
    tag = etree.QName(node)
    if tag.localname != 'block':
        node_elem = Element('block')
        node_elem.text = '{'
        for elem in case_elem.getparent()[index + 1:]:
            node_elem.append(elem)
            if elem.text == 'break;':
                elem_index = elem.getparent().index(elem)
                del elem.getparent()[elem_index]
                break
        node_elem.tail = '\n}'
        case_elem.getparent().insert(index + 1, node_elem)


# judge whether switch can be transformed
def judge_transform(switch_elem):
    global flag_switch
    # judge default
    default_elem = get_default(switch_elem)
    if len(default_elem) > 0:
        index = default_elem[0].getparent().index(default_elem[0])
        if len(default_elem[0].getparent()) <= index + 1:
            flag_switch = False
            return
        node = default_elem[0].getparent()[index + 1]
        tag = etree.QName(node)
        if tag.localname == 'block':
            if len(node) != 0 and len(node[0]) != 0 and node[0][-1].text != 'break;':
                flag_switch = False
                return
        else:
            if default_elem[0].getparent()[-1].text != 'break;':
                flag_switch = False
                return

    # judge case
    case_elems = get_case(switch_elem)
    if len(case_elems) > 0:
        for case_elem in case_elems:
            index = case_elem.getparent().index(case_elem)
            if len(case_elem.getparent()) <= index + 1:
                flag_switch = False
                return
            node = case_elem.getparent()[index + 1]
            tag = etree.QName(node)
            if tag.localname == 'block':
                if len(node) > 0 and len(node[0]) > 0 and node[0][-1].text != 'break;':
                    flag_switch = False
                    return
            else:
                for elem in case_elem.getparent()[index + 1:]:
                    tag = etree.QName(elem)
                    if tag.localname != 'break':
                        flag_switch = False
                    if tag.localname == 'break':
                        flag_switch = True
                        break
                    if tag.localname == 'case' or tag.localname == 'default':
                        flag_switch = False
                        return
    else:
        flag_switch = False


# start to transform
def trans_tree(e):
    global flag
    global flag_switch
    # get all swtich
    switch_elems = get_switch(e)
    for switch_elem in switch_elems:
        # judge whether switch can be transformed
        flag_switch = True
        judge_transform(switch_elem)
        if flag_switch is False:
            continue
        # get all case
        case_elems = get_case(switch_elem)
        # get the conditional variable in switch
        var = get_variable(switch_elem)[0]
        var[-1].tail = ''
        # get default
        default_elem = get_default(switch_elem)
        # default sentence starts to transform
        transform_default(default_elem)
        # every case starts to transform
        transform_case(case_elems, var)
        var_index = var.getparent().index(var)
        del var.getparent()[var_index]
        # get all the execution sentences in switch and move to the front of switch
        block_content = get_block_content(switch_elem)[0]
        block_content.tail = ''
        get_switch_index = switch_elem.getparent().index(switch_elem)
        switch_elem.getparent().insert(get_switch_index, block_content)
        # delete switch
        index = switch_elem.getparent().index(switch_elem)
        del switch_elem.getparent()[index]
        flag = True


# save tree ti file
def save_tree_to_file(tree, path):
    with open(path, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


def count(e):
    count_num = 0
    global flag_switch
    # get all switch
    switch_elems = get_switch(e)
    for switch_elem in switch_elems:
        flag_switch = True
        judge_transform(switch_elem)
        if flag_switch is True:
            count_num += 1
    return count_num


# calculate the number of switch in the program
def get_number(xml_path):
    xml_file_path = os.path.abspath(xml_path)
    e = init_parse(xml_file_path)
    return count(e)


def etree_transform(evaluator, dst_style: str):
    trans_tree(evaluator)
    return evaluator


# program's input port
def program_transform(program_path, output_xml_path: str = './style/style.xml'):
    e = init_parse(program_path)
    trans_tree(e)
    save_tree_to_file(doc, output_xml_path)
