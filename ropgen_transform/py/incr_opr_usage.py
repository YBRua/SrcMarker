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


def get_expr_stmts(e):
    return e('//src:expr_stmt')


def get_expr(elem):
    return elem.xpath('src:expr', namespaces=ns)


def get_operator(elem):
    return elem.xpath('src:operator', namespaces=ns)


def get_for_incrs(e):
    return e('//src:for/src:control/src:incr/src:expr')


def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


def get_standalone_exprs(e):
    standalone_exprs = []
    #get all expression statements
    expr_stmts = get_expr_stmts(e)
    for expr_stmt in expr_stmts:
        expr = get_expr(expr_stmt)
        #there should be exactly one expression in a statement
        if len(expr) != 1: continue
        standalone_exprs.append(expr[0])
    return standalone_exprs


#type - 1: standalone statements, 2: for increments, 3: both
def get_incr_exprs(e, type):
    incr_exprs = []
    if type == 1:
        exprs = get_standalone_exprs(e)
    elif type == 2:
        exprs = get_for_incrs(e)
    elif type == 3:
        exprs = get_standalone_exprs(e) + get_for_incrs(e)
    for expr in exprs:
        opr = get_operator(expr)
        #exactly one operator, which should be ++/-- or +=/-=
        if len(opr) == 1:
            if opr[0].text == '++' or opr[0].text == '--':
                if opr[0].getparent().index(opr[0]) == 0:
                    incr_exprs.append((expr, 2))
                else:
                    incr_exprs.append((expr, 1))
            elif opr[0].text == '+=' or opr[0].text == '-=':
                token_after_opr = opr[0].getnext()
                if token_after_opr.text == '1':
                    incr_exprs.append((expr, 4))
        # two operators, e.g. i=i+1
        elif len(opr) == 2:
            if opr[0].text == '=' and (opr[1].text == '+' or opr[1].text == '-'):
                token_before_opr0 = opr[0].getprevious()
                token_after_opr0 = opr[0].getnext()
                token_after_opr1 = opr[1].getnext()
                if token_after_opr1.text == '1':
                    if ''.join(token_before_opr0.itertext()) == ''.join(
                            token_after_opr0.itertext()):
                        incr_exprs.append((expr, 3))
    #print(incr_exprs)
    return incr_exprs


# i+=1/i-=1 to i++/i--
def separate_incr_to_incr_postfix(opr):
    token_after_opr = opr[0].getnext()
    if token_after_opr is not None:
        if token_after_opr.getparent() is not None:
            if opr[0].text == '+=':
                opr[0].text = '++'
                token_after_opr.getparent().remove(token_after_opr)
            elif opr[0].text == '-=':
                opr[0].text = '--'
                token_after_opr.getparent().remove(token_after_opr)


# i+=1/i-=1 to ++i/--i
def separate_incr_to_incr_prefix(opr):
    token_before_opr = opr[0].getprevious()
    token_after_opr = opr[0].getnext()
    if token_after_opr is not None:
        if opr[0].text == '+=':
            token_before_opr.text = '++' + token_before_opr.text
            token_after_opr.getparent().remove(token_after_opr)
        elif opr[0].text == '-=':
            token_before_opr.text = '--' + token_before_opr.text
            token_after_opr.getparent().remove(token_after_opr)


# i++/++i/i--/--i to i+=1/i-=1
def incr_to_separate_incr(opr, expr):
    if opr[0].text == '++':
        opr[0].getparent().remove(opr[0])
        if expr.tail is not None:
            expr.tail = ' += 1' + expr.tail
        else:
            expr.tail = ' += 1'
    elif opr[0].text == '--':
        opr[0].getparent().remove(opr[0])
        if expr.tail is not None:
            expr.tail = ' -= 1' + expr.tail
        else:
            expr.tail = ' -= 1'


# i=i+1/i=i-1 to i+=1/i-=1
def full_incr_to_separate_incr(opr, expr):
    operator = opr[1].text
    if operator == '+':
        del expr[2:4]
        opr[0].text = '+='
    elif operator == '-':
        del expr[2:4]
        opr[0].text = '-='


# the above reversed
def separate_incr_to_full_incr(opr, expr):
    operator = opr[0].text
    token_before_opr = opr[0].getprevious()
    if operator == '+=':
        opr[0].text = '= ' + token_before_opr.text + ' + 1'
    elif operator == '-=':
        opr[0].text = '= ' + token_before_opr.text + ' - 1'


# i=i+1/i=i-1 to i++/++i/i--/--i
# 'pre_or_post' indicates whether target style is prefixed (e.g. ++i) or postfixed (e.g. i++)
# 0- pre, 1- post
def full_incr_to_incr(opr, expr, pre_or_post):
    operator = opr[1].text
    if expr[0].text is not None and expr[-1].text == '1':
        if operator == '+':
            del expr[1:]
            if pre_or_post == 1:
                if expr.tail is not None:
                    expr.tail = '++' + expr.tail
                else:
                    expr.tail = '++'
            else:
                expr[0].text = '++' + expr[0].text
        elif operator == '-':
            del expr[1:]
            if pre_or_post == 1:
                if expr.tail is not None:
                    expr.tail = '--' + expr.tail
                else:
                    expr.tail = '--'
            else:
                expr[0].text = '--' + expr[0].text


# the above reversed
# 'pre_or_post' indicates whether target style is prefixed (e.g. ++i) or postfixed (e.g. i++)
# 0- pre, 1- post
def incr_to_full_incr(opr, expr, pre_or_post):
    operator = opr[0].text
    #print(pre_or_post)
    var_name = ''.join(expr[int(not pre_or_post)].itertext())
    if var_name == '++' or var_name == '--': return
    if expr[int(not pre_or_post)].text is None: return
    if operator == '++':
        del expr[pre_or_post]
        expr[0].text = var_name + ' = ' + var_name + ' + 1'
    elif operator == '--':
        del expr[pre_or_post]
        expr[0].text = var_name + ' = ' + var_name + ' - 1'


def transform_standalone_stmts(e):
    global flag
    for expr, style in get_incr_exprs(e, 1):
        opr = get_operator(expr)
        if style == 1 or style == 2:
            incr_to_separate_incr(opr, expr)
        elif style == 4:
            separate_incr_to_incr_postfix(opr)


def transform_for_loops(e):
    for expr, style in get_incr_exprs(e, 2):
        opr = get_operator(expr)
        if style == 1 or style == 2:
            incr_to_separate_incr(opr, expr)
        elif style == 4:
            separate_incr_to_incr_postfix(opr)


def xml_file_path(xml_path):
    global flag
    # xml_path XML path to be converted
    # sub_dir_list Package name for each author
    # name_list Specific XML file name
    save_xml_file = './transform_xml_file/incr_opr_usage'
    transform_java_file = './target_author_file/transform_java/incr_opr_usage'
    if not os.path.exists(transform_java_file):
        os.mkdir(transform_java_file)
    if not os.path.exists(save_xml_file):
        os.mkdir(save_xml_file)
    for xml_path_elem in xml_path:
        xmlfilepath = os.path.abspath(xml_path_elem)
        e = init_parser(xmlfilepath)
        flag = False
        transform_standalone_stmts(e)
        if flag == True:
            str = xml_path_elem.split('/')[-1]
            sub_dir = xml_path_elem.split('/')[-2]
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_tree_to_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file


if __name__ == '__main__':
    e = init_parser(sys.argv[1])
    transform_standalone_stmts(e)
    #transform_for_loops(e)
    #print(get_program_style(e))
    save_tree_to_file(doc, './incr_opr.xml')
