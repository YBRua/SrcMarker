"""
    ternary operation structure -> if structure

"""
import os
from lxml import etree
from lxml.etree import Element

doc = None
ns = {'src': 'http://www.srcML.org/srcML/src'}
str = '{http://www.srcML.org/srcML/src}'
flag = True  # judge whether the transformation is successful


def init_parse(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc, namespaces={'src': 'http://www.srcML.org/srcML/src'})
    return e


def get_ternary(e):
    return e('//src:ternary')


def get_condition(elem):
    return elem.xpath('src:condition', namespaces=ns)


# executable statement when ternary operation condition is true
def get_expr1(elem):
    return elem.xpath('src:then/src:expr', namespaces=ns)


# executable statement when ternary operation condition if false
def get_expr2(elem):
    return elem.xpath('src:else/src:expr', namespaces=ns)


# start transform
def trans_tree(e):
    global flag
    # get all ternary
    ternary_elems = get_ternary(e)
    for ternary_elem in ternary_elems:
        # judge whether it belongs to ternary operation
        if ternary_elem.getparent().tag == str + 'expr' and ternary_elem.getparent(
        ).getparent().tag == str + 'expr_stmt':
            index = ternary_elem.getparent().index(ternary_elem)
            if (ternary_elem.getparent()[index - 1].text
                    == '=') or (ternary_elem.getparent()[index - 1].text == '('
                                and ternary_elem.getparent()[index - 2].text == '='):
                # get ternary's condition
                condition = get_condition(ternary_elem)[0]
                if condition[0][0].text != '(' or condition[0][-1].text != ')':
                    condition.text = '('
                    condition[0][-1].tail = ')'
                condition[0].tail = ''
                # get variable and '='
                expr = ternary_elem.getparent()
                expr_node = Element('expr')
                for expr_e in expr:
                    if expr_e.tag == str + 'ternary':
                        break
                    expr_node.append(expr_e)
                text = ''.join(expr_node.itertext()).replace('\n', '').replace(' ', '')
                text_list = list(text)
                if text_list[-1] == '(':
                    del text_list[-1]
                    text = ''.join(text_list)
                expr_node1 = Element('expr')
                expr_node2 = Element('expr')
                expr_node1.text = text
                expr_node2.text = text
                # get expression1 and expression2
                elem_1 = get_expr1(ternary_elem)[0]
                elem_1.tail = ';\n}'
                if len(get_expr2(ternary_elem)) == 0: continue
                elem_2 = get_expr2(ternary_elem)[0]

                elem_2.tail = ';\n}'
                # build if structure
                if_node = Element('if')
                if_node.text = 'if'
                # add if's condition
                if_node.append(condition)
                # add if's executable sentence
                if_block = Element('block')
                if_block.text = '{'
                if_node.append(if_block)
                # add expression 1 to the statement where the if condition is true
                if_block.append(expr_node1)
                if_block.append(elem_1)
                # build else structure
                else_node = Element('else')
                else_node.text = 'else'
                # add else's executable sentence
                else_block = Element('block')
                else_block.text = '{'
                else_node.append(else_block)
                # add expression 2 to the statement where the if condition is false(else sentence)
                else_block.append(expr_node2)
                else_block.append(elem_2)
                expr_stmt = expr.getparent()
                index = expr_stmt.getparent().index(expr_stmt)
                # insert the if statement after the ternary operation
                expr_stmt.getparent().insert(index + 1, if_node)
                expr_stmt.getparent().insert(index + 2, else_node)

                # delete ternary statement
                del expr_stmt.getparent()[index]
                flag = True


def save_tree_to_file(tree, path):
    with open(path, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


def count(e):
    count_num = 0
    # get all ternary operation
    ternary_elems = get_ternary(e)
    for ternary_elem in ternary_elems:
        # judge whether it belongs to ternary operation
        if ternary_elem.getparent().tag == str + 'expr' and ternary_elem.getparent(
        ).getparent().tag == str + 'expr_stmt':
            index = ternary_elem.getparent().index(ternary_elem)
            if (ternary_elem.getparent()[index - 1].text
                    == '=') or (ternary_elem.getparent()[index - 1].text == '('
                                and ternary_elem.getparent()[index - 2].text == '='):
                count_num += 1
    return count_num


# calculate the number of ternary in the program
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
