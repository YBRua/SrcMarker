import os
from lxml import etree

ns = {"src": "http://www.srcML.org/srcML/src"}
doc = None
flag = True


def init_parse(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc, namespaces={"src": "http://www.srcML.org/srcML/src"})
    return e


def get_expr(e):
    return e("//src:expr")


def get_expr_stmt(e):
    return e("//src:expr_stmt/src:expr")


def expr_transform(expr_elem, tree_root, expr_elem_prev_path, new_ignore_list):
    global flag
    flag = False
    # temp = i; i++;
    if len(expr_elem) == 3:
        index = expr_elem.getparent().index(expr_elem)
        parent = expr_elem.getparent()
        if len(parent) > index + 1:
            tag = etree.QName(parent[index + 1])
            if tag.localname == "expr":
                if len(parent[index + 1]) == 2:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1]
                    if expr_1[1].text == "=":
                        if len(expr_1[2]) == len(expr_2[0]) and len(expr_2[0]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[0])):
                                if expr_1[2][i].text != expr_2[0][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "++" or expr_2[1].text == "--":
                                    expr_1.append(expr_2[1])
                                    del parent[index + 1]
                                    flag = True

                                    new_ignore_list.append(expr_elem_prev_path)

                        elif len(expr_2[0]) == 0 and expr_1[2].text == expr_2[0].text:
                            if expr_2[1].text == "++" or expr_2[1].text == "--":
                                expr_1.append(expr_2[1])
                                del parent[index + 1]
                                flag = True

                                new_ignore_list.append(expr_elem_prev_path)
    # ++i; temp = i;
    if len(expr_elem) == 2:
        index = expr_elem.getparent().index(expr_elem)
        parent = expr_elem.getparent()
        if len(parent) > index + 1:
            tag = etree.QName(parent[index + 1])
            if tag.localname == "expr":
                if len(parent[index + 1]) == 3:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1]
                    if expr_1[0].text == "++" or expr_1[0].text == "--":
                        if len(expr_1[1]) == len(expr_2[2]) and len(expr_1[1]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[2])):
                                if expr_1[1][i].text != expr_2[2][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "=":
                                    expr_2.replace(expr_2[2], expr_1)
                                    expr_1.tail = ""
                                    flag = True

                                    expr_2_pre = expr_2.getparvious()
                                    expr_2_pre = (
                                        expr_2_pre if expr_2_pre is not None else expr_2
                                    )
                                    expr_2_pre_path = tree_root.getpath(expr_2_pre)
                                    new_ignore_list.append(expr_2_pre_path)

                        elif len(expr_2[2]) == 0 and expr_1[1].text == expr_2[2].text:
                            if expr_2[1].text == "=":
                                expr_2.replace(expr_2[2], expr_1)
                                expr_1.tail = ""
                                flag = True

                                expr_2_pre = expr_2.getparvious()
                                expr_2_pre = (
                                    expr_2_pre if expr_2_pre is not None else expr_2
                                )
                                expr_2_pre_path = tree_root.getpath(expr_2_pre)
                                new_ignore_list.append(expr_2_pre_path)

    return flag, tree_root, new_ignore_list


def expr_stmt_transfrom(
    expr_elem, tree_root, expr_elem_stmt_prev_path, new_ignore_list
):
    global flag
    flag = False
    # temp = i; i++;
    if len(expr_elem) == 3:
        parent_expr_stmt = expr_elem.getparent()
        index = parent_expr_stmt.getparent().index(parent_expr_stmt)
        parent = parent_expr_stmt.getparent()
        if len(parent) > index + 1:
            tag = etree.QName(parent[index + 1])
            if tag.localname == "expr_stmt":
                if len(parent[index + 1]) > 0 and len(parent[index + 1][0]) == 2:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1][0]
                    if expr_1[1].text == "=":
                        if len(expr_2[0]) == len(expr_1[2]) and len(expr_2[0]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[0])):
                                if expr_1[2][i].text != expr_2[0][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "++" or expr_2[1].text == "--":
                                    expr_1.append(expr_2[1])
                                    del parent[index + 1]
                                    flag = True

                                    new_ignore_list.append(expr_elem_stmt_prev_path)

                        elif len(expr_2[0]) == 0 and expr_2[0].text == expr_1[2].text:
                            if expr_2[1].text == "++" or expr_2[1].text == "--":
                                expr_1.append(expr_2[1])
                                del parent[index + 1]
                                flag = True
                                new_ignore_list.append(expr_elem_stmt_prev_path)
            if tag.localname == "expr":
                if len(parent[index + 1]) == 2:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1]
                    if expr_1[1].text == "=":
                        if len(expr_2[0]) == len(expr_1[2]) and len(expr_2[0]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[0])):
                                if expr_1[2][i].text != expr_2[0][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "++" or expr_2[1].text == "--":
                                    expr_1.append(expr_2[1])
                                    del parent[index + 1]
                                    flag = True
                                    #
                                    new_ignore_list.append(expr_elem_stmt_prev_path)
                        elif len(expr_2[0]) == 0 and expr_2[0].text == expr_1[2].text:
                            if expr_2[1].text == "++" or expr_2[1].text == "--":
                                expr_1.append(expr_2[1])
                                del parent[index + 1]
                                flag = True
                                #
                                new_ignore_list.append(expr_elem_stmt_prev_path)
    # ++i; temp = i;
    if len(expr_elem) == 2:
        parent_expr_stmt = expr_elem.getparent()
        if parent_expr_stmt.getparent() is not None:
            index = parent_expr_stmt.getparent().index(parent_expr_stmt)
            parent = parent_expr_stmt.getparent()
            if len(parent) > index + 1:
                tag = etree.QName(parent[index + 1])
                if tag.localname == "expr_stmt":
                    if len(parent[index + 1]) > 0 and len(parent[index + 1][0]) == 3:
                        expr_1 = expr_elem
                        expr_2 = parent[index + 1][0]
                        if expr_1[0].text == "++" or expr_1[0].text == "--":
                            if len(expr_1[1]) == len(expr_2[2]) and len(expr_1[1]) > 0:
                                flag_expr = True
                                for i in range(0, len(expr_2[2])):
                                    if expr_1[1][i].text != expr_2[2][i].text:
                                        flag_expr = False
                                if flag_expr == True:
                                    if expr_2[1].text == "=":
                                        expr_2.replace(expr_2[2], expr_1)
                                        expr_1.tail = ""
                                        flag = True
                                        #
                                        new_ignore_list.append(expr_elem_stmt_prev_path)
                            elif (
                                len(expr_2[2]) == 0 and expr_1[1].text == expr_2[2].text
                            ):
                                if expr_2[1].text == "=":
                                    expr_2.replace(expr_2[2], expr_1)
                                    expr_1.tail = ""
                                    flag = True
                                    #
                                    new_ignore_list.append(expr_elem_stmt_prev_path)
                if tag.localname == "expr":
                    if len(parent[index + 1]) == 3:
                        expr_1 = expr_elem
                        expr_2 = parent[index + 1]
                        if expr_1[0].text == "++" or expr_1[0].text == "--":
                            if len(expr_1[1]) == len(expr_2[2]) and len(expr_1[1]) > 0:
                                flag_expr = True
                                for i in range(0, len(expr_2[2])):
                                    if expr_1[1][i].text != expr_2[2][i].text:
                                        flag_expr = False
                                if flag_expr == True:
                                    if expr_2[1].text == "=":
                                        expr_2.replace(expr_2[2], expr_1)
                                        expr_1.tail = ""
                                        flag = True
                                        #
                                        new_ignore_list.append(expr_elem_stmt_prev_path)
                            elif (
                                len(expr_2[2]) == 0 and expr_1[1].text == expr_2[2].text
                            ):
                                if expr_2[1].text == "=":
                                    expr_2.replace(expr_2[2], expr_1)
                                    expr_1.tail = ""
                                    flag = True
                                    #
                                    new_ignore_list.append(expr_elem_stmt_prev_path)
    return flag, tree_root, new_ignore_list


def trans_tree(e, ignore_list=[], instances=None):
    # 得到所有的expr
    tree_root = e("/*")[0].getroottree()
    new_ignore_list = []

    expr_elems = [
        get_expr(e) if instances is None else (instance[0] for instance in instances)
    ]
    expr_stmt_elems = [
        get_expr_stmt(e)
        if instances is None
        else (instance[0] for instance in instances)
    ]
    for item in expr_elems:
        for expr_elem in item:
            expr_elem_prev = expr_elem.getprevious()
            expr_elem_prev = expr_elem_prev if expr_elem_prev is not None else expr_elem
            expr_elem_prev_path = tree_root.getpath(expr_elem_prev)
            if expr_elem_prev_path in ignore_list:
                continue

            expr_transform(expr_elem, tree_root, expr_elem_prev_path, new_ignore_list)

    for item in expr_stmt_elems:
        for expr_stmt_elem in item:
            expr_stmt_elem_prev = expr_stmt_elem.getparent().getprevious()
            expr_stmt_elem_prev = (
                expr_stmt_elem_prev
                if expr_stmt_elem_prev is not None
                else expr_stmt_elem.getparent()
            )
            expr_stmt_elem_prev_path = tree_root.getpath(expr_stmt_elem_prev)
            if expr_stmt_elem_prev_path in ignore_list:
                continue

            expr_stmt_transfrom(
                expr_stmt_elem, tree_root, expr_stmt_elem_prev_path, new_ignore_list
            )


def expr_transform_count(expr_elem, count_num):
    # temp = i; i++;
    if len(expr_elem) == 3:
        index = expr_elem.getparent().index(expr_elem)
        parent = expr_elem.getparent()
        if len(parent) > index + 1:
            tag = etree.QName(parent[index + 1])
            if tag.localname == "expr":
                if len(parent[index + 1]) == 2:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1]
                    if expr_1[1].text == "=":
                        if len(expr_1[2]) == len(expr_2[0]) and len(expr_2[0]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[0])):
                                if expr_1[2][i].text != expr_2[0][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "++" or expr_2[1].text == "--":
                                    count_num += 1
                        elif len(expr_2[0]) == 0 and expr_1[2].text == expr_2[0].text:
                            if expr_2[1].text == "++" or expr_2[1].text == "--":
                                count_num += 1
    # ++i; temp = i;
    if len(expr_elem) == 2:
        index = expr_elem.getparent().index(expr_elem)
        parent = expr_elem.getparent()
        if len(parent) > index + 1:
            tag = etree.QName(parent[index + 1])
            if tag.localname == "expr":
                if len(parent[index + 1]) == 3:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1]
                    if expr_1[0].text == "++" or expr_1[0].text == "--":
                        if len(expr_1[1]) == len(expr_2[2]) and len(expr_1[1]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[2])):
                                if expr_1[1][i].text != expr_2[2][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "=":
                                    count_num += 1
                        elif len(expr_2[2]) == 0 and expr_1[1].text == expr_2[2].text:
                            if expr_2[1].text == "=":
                                count_num += 1
    return count_num


def expr_stmt_transfrom_count(expr_elem, count_num):
    # temp = i; i++;
    if len(expr_elem) == 3:
        parent_expr_stmt = expr_elem.getparent()
        index = parent_expr_stmt.getparent().index(parent_expr_stmt)
        parent = parent_expr_stmt.getparent()
        if len(parent) > index + 1:
            tag = etree.QName(parent[index + 1])
            if tag.localname == "expr_stmt":
                if len(parent[index + 1]) > 0 and len(parent[index + 1][0]) == 2:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1][0]
                    if expr_1[1].text == "=":
                        if len(expr_2[0]) == len(expr_1[2]) and len(expr_2[0]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[0])):
                                if expr_1[2][i].text != expr_2[0][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "++" or expr_2[1].text == "--":
                                    count_num += 1
                        elif len(expr_2[0]) == 0 and expr_2[0].text == expr_1[2].text:
                            if expr_2[1].text == "++" or expr_2[1].text == "--":
                                count_num += 1
            if tag.localname == "expr":
                if len(parent[index + 1]) == 2:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1]
                    if expr_1[1].text == "=":
                        if len(expr_2[0]) == len(expr_1[2]) and len(expr_2[0]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[0])):
                                if expr_1[2][i].text != expr_2[0][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "++" or expr_2[1].text == "--":
                                    count_num += 1
                        elif len(expr_2[0]) == 0 and expr_2[0].text == expr_1[2].text:
                            if expr_2[1].text == "++" or expr_2[1].text == "--":
                                count_num += 1
    # ++i; temp = i;
    if len(expr_elem) == 2:
        parent_expr_stmt = expr_elem.getparent()
        index = parent_expr_stmt.getparent().index(parent_expr_stmt)
        parent = parent_expr_stmt.getparent()
        if len(parent) > index + 1:
            tag = etree.QName(parent[index + 1])
            if tag.localname == "expr_stmt":
                if len(parent[index + 1]) > 0 and len(parent[index + 1][0]) == 3:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1][0]
                    if expr_1[0].text == "++" or expr_1[0].text == "--":
                        if len(expr_1[1]) == len(expr_2[2]) and len(expr_1[1]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[2])):
                                if expr_1[1][i].text != expr_2[2][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "=":
                                    count_num += 1
                        elif len(expr_2[2]) == 0 and expr_1[1].text == expr_2[2].text:
                            if expr_2[1].text == "=":
                                count_num += 1
            if tag.localname == "expr":
                if len(parent[index + 1]) == 3:
                    expr_1 = expr_elem
                    expr_2 = parent[index + 1]
                    if expr_1[0].text == "++" or expr_1[0].text == "--":
                        if len(expr_1[1]) == len(expr_2[2]) and len(expr_1[1]) > 0:
                            flag_expr = True
                            for i in range(0, len(expr_2[2])):
                                if expr_1[1][i].text != expr_2[2][i].text:
                                    flag_expr = False
                            if flag_expr == True:
                                if expr_2[1].text == "=":
                                    count_num += 1
                        elif len(expr_2[2]) == 0 and expr_1[1].text == expr_2[2].text:
                            if expr_2[1].text == "=":
                                count_num += 1
    return count_num


def save_tree_to_file(tree, file):
    with open(file, "w") as f:
        f.write(etree.tostring(tree).decode("utf8"))


def count(e):
    count_num = 0
    expr_elems = get_expr(e)
    expr_stmt_elems = get_expr_stmt(e)
    for expr_elem in expr_elems:
        count_num = expr_transform_count(expr_elem, count_num)
    for expr_stmt_elem in expr_stmt_elems:
        count_num = expr_stmt_transfrom_count(expr_stmt_elem, count_num)
    return count_num


def get_number(xml_path):
    xmlfilepath = os.path.abspath(xml_path)
    e = init_parse(xmlfilepath)
    return count(e)


def program_transform(input_xml_path: str, output_xml_path: str = "./style/style.xml"):
    e = init_parse(input_xml_path)
    trans_tree(e)
    save_tree_to_file(doc, output_xml_path)
