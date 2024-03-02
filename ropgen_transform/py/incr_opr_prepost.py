import sys
from lxml import etree
from . import incr_opr_usage
from ropgen_transform.xml_utils import init_parser, load_doc, XML_NS


def get_expr_stmts(e):
    return e("//src:expr_stmt")


def get_expr(elem):
    return elem.xpath("src:expr", namespaces=XML_NS)


def get_operator(elem):
    return elem.xpath("src:operator", namespaces=XML_NS)


def get_for_incrs(e):
    return e("//src:for/src:control/src:incr/src:expr")


def save_tree_to_file(tree, file):
    with open(file, "w") as f:
        f.write(etree.tostring(tree).decode("utf8"))


def get_standalone_exprs(e):
    standalone_exprs = []
    # get all expression statements
    expr_stmts = get_expr_stmts(e)
    for expr_stmt in expr_stmts:
        expr = get_expr(expr_stmt)
        # there should be exactly one expression in a statement
        if len(expr) != 1:
            continue
        standalone_exprs.append(expr[0])
    return standalone_exprs


# not used
def transform_standalone_stmts(e):
    global flag
    exprs = get_standalone_exprs(e)
    for expr in exprs:
        opr = get_operator(expr)
        # and exactly one operator, which should be ++ or --
        if len(opr) == 1:
            if opr[0].text == "++":
                flag = True
                if opr[0].getparent().index(opr[0]) == 0:
                    opr[0].getparent().remove(opr[0])
                    expr.tail = "++;"
                else:
                    opr[0].getparent().remove(opr[0])
                    expr.text = "++"
            elif opr[0].text == "--":
                flag = True
                if opr[0].getparent().index(opr[0]) == 0:
                    opr[0].getparent().remove(opr[0])
                    expr.tail = "--;"
                else:
                    opr[0].getparent().remove(opr[0])
                    expr.text = "--"


# not used
def transform_for_loops(e):
    for incr in get_for_incrs(e):
        opr = get_operator(incr)
        if len(opr) == 1:
            if opr[0].text == "++":
                if opr[0].getparent().index(opr[0]) == 0:
                    opr[0].getparent().remove(opr[0])
                    incr.tail = "++;"
                else:
                    opr[0].getparent().remove(opr[0])
                    incr.text = "++"
            elif opr[0].text == "--":
                if opr[0].getparent().index(opr[0]) == 0:
                    opr[0].getparent().remove(opr[0])
                    incr.tail = "--;"
                else:
                    opr[0].getparent().remove(opr[0])
                    incr.text = "--"


# entry and dispatcher function
# actual code that does the transformation is in incr_opr_usage.py (except for style 10.1 to/from 10.2, which are in this function)
# arguments 'ignore_list' and 'instances' are pretty much legacy code and can be ignored
# argument 'e' is obtained by calling init_parser(srcML XML path)
# 'src_style' the style of source author
# 'dst_style' the style of target author
def transform(evaluator, src_style, dst_style):
    incr_exprs = [get_standalone_exprs(evaluator)]
    tree_root = evaluator("/*")[0].getroottree()

    src_dst_tuple = (src_style, dst_style)
    for item in incr_exprs:
        for incr_expr in item:
            incr_expr_grandparent = incr_expr.getparent().getparent()
            if incr_expr_grandparent is None:
                return evaluator

            opr = get_operator(incr_expr)
            if len(opr) == 1:
                if opr[0].text == "++":
                    if src_dst_tuple == ("10.2", "10.1"):
                        opr[0].getparent().remove(opr[0])
                        new_opr = etree.Element("operator")
                        new_opr.text = "++"
                        incr_expr.append(new_opr)
                    elif src_dst_tuple == ("10.1", "10.2"):
                        opr[0].getparent().remove(opr[0])
                        incr_expr.text = "++"
                    elif src_dst_tuple == ("10.1", "10.4"):
                        incr_opr_usage.incr_to_separate_incr(opr, incr_expr)
                    elif src_dst_tuple == ("10.2", "10.4"):
                        incr_opr_usage.incr_to_separate_incr(opr, incr_expr)
                    elif src_dst_tuple == ("10.4", "10.1"):
                        incr_opr_usage.separate_incr_to_incr_postfix(opr)
                    elif src_dst_tuple == ("10.4", "10.2"):
                        incr_opr_usage.separate_incr_to_incr_prefix(opr)
                    elif src_dst_tuple == ("10.1", "10.3"):
                        incr_opr_usage.incr_to_full_incr(opr, incr_expr, 1)
                    elif src_dst_tuple == ("10.2", "10.3"):
                        incr_opr_usage.incr_to_full_incr(opr, incr_expr, 0)
                    elif src_dst_tuple == ("10.4", "10.3"):
                        incr_opr_usage.separate_incr_to_full_incr(opr, incr_expr)
                elif opr[0].text == "--":
                    if src_dst_tuple == ("10.2", "10.1"):
                        opr[0].getparent().remove(opr[0])
                        new_opr = etree.Element("operator")
                        new_opr.text = "--"
                        incr_expr.append(new_opr)
                    elif src_dst_tuple == ("10.1", "10.2"):
                        opr[0].getparent().remove(opr[0])
                        incr_expr.text = "--"
                    elif src_dst_tuple == ("10.1", "10.4"):
                        incr_opr_usage.incr_to_separate_incr(opr, incr_expr)
                    elif src_dst_tuple == ("10.2", "10.4"):
                        incr_opr_usage.incr_to_separate_incr(opr, incr_expr)
                    elif src_dst_tuple == ("10.4", "10.1"):
                        incr_opr_usage.separate_incr_to_incr_postfix(opr)
                    elif src_dst_tuple == ("10.4", "10.2"):
                        incr_opr_usage.separate_incr_to_incr_prefix(opr)
            elif len(opr) == 2:
                token_before_opr0 = opr[0].getprevious()
                token_after_opr0 = opr[0].getnext()
                token_after_opr1 = opr[1].getnext()
                if (
                    token_before_opr0 is not None
                    and token_after_opr0 is not None
                    and token_after_opr1 is not None
                ):
                    if token_after_opr1.text == "1" and "".join(
                        token_before_opr0.itertext()
                    ) == "".join(token_after_opr0.itertext()):
                        if src_dst_tuple == ("10.3", "10.1"):
                            incr_opr_usage.full_incr_to_incr(opr, incr_expr, 1)
                        elif src_dst_tuple == ("10.3", "10.2"):
                            incr_opr_usage.full_incr_to_incr(opr, incr_expr, 0)
                        elif src_dst_tuple == ("10.3", "10.4"):
                            incr_opr_usage.full_incr_to_separate_incr(opr, incr_expr)
    return evaluator


def transform_all(evaluator, dst_style: str):
    raise NotImplementedError("屎山代码。改不动了。")


def program_transform(
    program_path, style1, style2, output_xml_path="./style/style.xml"
):
    list1 = []
    instances = None
    e = init_parser(program_path)
    transform(e, style1, style2, list1, instances)
    save_tree_to_file(doc, output_xml_path)


def etree_transform(evaluator, dst_style: str):
    # TODO: current implementation is inefficient, should be optimized
    POSSIBLE_STYLES = ["10.1", "10.2", "10.3", "10.4"]
    for src_style in POSSIBLE_STYLES:
        if src_style == dst_style:
            continue
        evaluator = transform(evaluator, src_style, dst_style)
    return evaluator


def program_transform_all(program_path, dst_style, output_xml_path):
    doc = load_doc(program_path)
    e = init_parser(doc)
    etree_transform(e, dst_style)
    save_tree_to_file(doc, output_xml_path)


if __name__ == "__main__":
    doc = load_doc(sys.argv[1])
    e = init_parser(doc)
    transform(e, "10.4", "10.2")
    save_tree_to_file(doc, "./incr_opr.xml")
