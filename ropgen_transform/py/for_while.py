import os
from lxml import etree
from ropgen_transform.xml_utils import init_parser, load_doc, XML_NS


def get_for(e):
    return e('//src:for')


# get the initialization variable in the for loop
def get_init(elem):
    return elem.xpath('src:control/src:init', namespaces=XML_NS)


# get the initialization expression of for
def get_cond_expr(elem):
    return elem.xpath('src:control/src:condition/src:expr', namespaces=XML_NS)


# get the operation expression after the for loop
def get_incr(elem):
    return elem.xpath('src:control/src:incr', namespaces=XML_NS)


def get_oper_in_incr(elem):
    return elem.xpath('src:control/src:incr/src:operator', namespaces=XML_NS)


# get the content of the execution statement of for
def get_block(elem):
    return elem.xpath('src:block/src:block_content', namespaces=XML_NS)


# get the execution statement of for
def get_block1(elem):
    return elem.xpath('src:block', namespaces=XML_NS)


# save tree to file
def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


# start to transform
def trans_tree(e, ignore_list=[], instances=None):
    # Tag, indicating that for can be converted
    global flag
    flag = False
    # get all for
    for_stmts = [
        get_for(e) if instances is None else (instance[0] for instance in instances)
    ]
    # get the root of the tree
    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []
    for item in for_stmts:
        for for_stmt in item:
            # filter out the for in the last round
            # While to for does not insert a statement, so the part not changed is the previous statement for
            for_prev = for_stmt.getprevious()
            for_prev = for_prev if for_prev is not None else for_stmt
            # take the path to see if it is in the passed in ignore list
            for_prev_path = tree_root.getpath(for_prev)
            if for_prev_path in ignore_list:
                continue
            # Get the initialization part of for
            init = get_init(for_stmt)
            # Get loop conditions
            cond_expr = get_cond_expr(for_stmt)
            incr = get_incr(for_stmt)
            operator_in_incr = get_oper_in_incr(for_stmt)
            if len(init) >= 1 and len(cond_expr) >= 1 and len(incr) >= 1:
                flag = True
                # for each for, change the XML tag to while
                for_stmt.tag = 'while'
                for_stmt.text = 'while '
                init = init[0]
                cond_expr = cond_expr[0]
                incr = incr[0]
                # get the for execution statement
                if len(get_block(for_stmt)) == 0:
                    continue
                for_block = get_block(for_stmt)[0]
                for_block1 = get_block1(for_stmt)[0]
                if for_block1.text is None:
                    for_block1.text = '{'
                    for_block1.tail = '}'
                # if the initialization is empty, remove the ‘；’
                if len(init) == 0:
                    init.text = ''
                # semicolon after removing cyclic condition
                cond_expr.tail = ')'
                # because the loop increment has to be mentioned separately, semicolon and line feed are added after it
                if len(incr) >= 1:
                    incr.tail = ';\n'
                    # replace x++, y++ with x++; y++;
                    if len(operator_in_incr) >= 1:
                        for i in operator_in_incr:
                            if i.text == ',':
                                i.text = ';'
                else:
                    incr.tail = ''
                for_index = for_stmt.getparent().index(for_stmt)
                # insert the initialization statement before the for loop
                for_stmt.getparent().insert(for_index, init)
                for_block.append(incr)
                # record the position that has not changed after this transformation,
                # that is,the position of the for statement counting up two statements
                init_prev = init.getprevious()
                init_prev = init_prev if init_prev is not None else init
                init_prev_path = tree_root.getpath(init_prev)
                new_ignore_list.append(init_prev_path)
    return flag, tree_root, new_ignore_list


def count(e):
    count_num = 0
    for_stmts = get_for(e)
    for for_stmt in for_stmts:
        init = get_init(for_stmt)
        cond_expr = get_cond_expr(for_stmt)
        incr = get_incr(for_stmt)
        if len(init) >= 1 and len(cond_expr) >= 1 and len(incr) >= 1:
            count_num += 1
    return count_num


# alculate the number of for in the program
def get_number(xml_path):
    xmlfilepath = os.path.abspath(xml_path)
    e = init_parser(xmlfilepath)
    return count(e)


def program_transform(program_path, output_xml_path: str = './style/style.xml'):
    doc = load_doc(program_path)
    e = init_parser(doc)
    trans_tree(e)
    save_tree_to_file(doc, output_xml_path)


def etree_transform(evaluator):
    trans_tree(evaluator)
    return evaluator
