import os
import sys
from copy import deepcopy

from lxml import etree
from lxml.etree import Element

doc = None
flag = False
ns = {
    "src": "http://www.srcML.org/srcML/src",
    "cpp": "http://www.srcML.org/srcML/cpp",
    "pos": "http://www.srcML.org/srcML/position",
}

save_xml_file = "./transform_xml_file/temp_var_pre"
transform_java_file = "./target_author_file/transform_java/temp_var_pre"


def init_parse(e):
    global doc
    doc = etree.parse(e)
    e = etree.XPathEvaluator(doc, namespaces=ns)
    return e


def get_block_cons(e):
    return e("//src:block_content")


def get_init_temps(block_con):
    return block_con.xpath("src:decl_stmt/src:decl", namespaces=ns)


def get_exprs(elem):
    return elem.xpath("src:expr_stmt/src:expr/src:name", namespaces=ns)


def get_var_name(b):
    return b.xpath(".//src:name", namespaces=ns)


def judge_ini(var_index, block_con, var_name):
    f = False
    b = None
    index = var_index
    for b in block_con[var_index + 1 :]:
        all_name = get_var_name(b)
        for name in all_name:
            if len(name) != 0:
                continue
            if name.text == var_name:
                f = True
                break
        if f == True:
            index = block_con.index(b)
            break
    return index, f, b


def get_instances(e):
    instances = []
    block_cons = get_block_cons(e)
    for block_con in block_cons:
        ls_decl = []  # Store the variables defined by the header
        # Get the variables defined by the head
        for decl_stml in block_con:
            if decl_stml.tag != "{http://www.srcML.org/srcML/src}decl_stmt":
                break
            # If the start is decl_stmt, get all the decl that meet the requirements
            for decl in decl_stml:
                if len(decl) == 2 or len(decl) == 3:
                    modifier_num = decl[0].xpath("src:modifier", namespaces=ns)
                    if len(modifier_num) != 0:
                        first_mod_index = decl[0].index(modifier_num[0])
                        modifier = Element("modifier")
                        modifier.text = ""
                        for modi in decl[0][first_mod_index:]:
                            modifier.text += modi.text
                            decl[0].remove(modi)
                        modifier.text += " "
                        decl.insert(1, modifier)
                    ls_decl.append(decl)  # Get decl tags
        for decl in ls_decl:  # Add variable types to all decl
            typ = deepcopy(decl.getparent()[0][0])
            if len(decl[0]) == 0:
                decl.remove(decl.getchildren()[0])
                decl.insert(0, typ)

        for decl in ls_decl:
            var_index = block_con.index(decl.getparent())

            if len(decl.xpath("src:name", namespaces=ns)[0]) != 0:
                des_index, f, b_ele = judge_ini(
                    var_index,
                    block_con,
                    decl.xpath("src:name", namespaces=ns)[0][0].text,
                )
            else:
                des_index, f, b_ele = judge_ini(
                    var_index, block_con, decl.xpath("src:name", namespaces=ns)[0].text
                )

            decl_prev = block_con[des_index].getprevious()

            flag = True

            if f == True:
                instances.append((decl_prev, decl, block_con, des_index, b_ele))
            # elif decl.getparent().index(decl)!=1:
            #     decl.remove(decl[0])
    return instances


def trans_temp_var(e):
    # Only consider the temporary variables in the loop
    # and the condition body Get the <block_content> tag
    decls = [get_instances(e)]
    # Get all initial temporary variables in the statement block

    tree_root = e("/*")[0].getroottree()
    block_con = []
    for item in decls:
        for inst_tuple in item:
            decl = inst_tuple[1]
            block_con = inst_tuple[2]
            b_ele = inst_tuple[4]
            if decl is None or decl.tail is None:
                continue
            if (
                decl.tail.replace(" ", "").replace("\n", "") == ";"
                and len(decl.getparent()) != 1
            ):
                decl.getparent()[-2].tail = ";"
            if decl.tail != ";":
                decl.tail = ""
                decl[-1].tail = ";\n"
            block_con.insert(block_con.index(b_ele), decl)

    block_s = get_block_cons(e)
    for block_ in block_s:
        ls_dec = []  # Store the variables defined by the header
        for decl_stml in block_:
            if decl_stml.tag != "{http://www.srcML.org/srcML/src}decl_stmt":
                break
            # If the start is decl_stmt, get all the decl that meet the requirements
            for decl in decl_stml:
                if len(decl) == 2 or len(decl) == 3:
                    ls_dec.append(decl)
        for decl in ls_dec:
            if (
                decl.getparent().tag == "{http://www.srcML.org/srcML/src}decl_stmt"
                and decl.getparent().index(decl) != 0
            ):
                decl.remove(decl.getchildren()[0])

    return tree_root


def save_file(doc, param):
    with open(param, "w") as f:
        f.write(etree.tostring(doc).decode("utf-8"))


def get_style(xmlfilepath):
    e = init_parse(xmlfilepath)

    num = 0
    block_cons = get_block_cons(e)
    for block_con in block_cons:
        ls_decl = []
        for decl_stml in block_con:
            if decl_stml.tag != "{http://www.srcML.org/srcML/src}decl_stmt":
                break
            for decl in decl_stml:
                if len(decl) == 2 or (
                    len(decl) == 3
                    and len(decl[2]) > 0
                    and len(decl[2][0]) > 0
                    and decl[2][0][0].tag == "{http://www.srcML.org/srcML/src}literal"
                ):
                    ls_decl.append(decl)  #
        for decl in ls_decl:  #
            typ = deepcopy(decl.getparent()[0][0])
            if len(decl[0]) == 0:
                pass

        for decl in ls_decl:
            var_index = block_con.index(decl.getparent())
            des_index, f, b = judge_ini(var_index, block_con, decl[1].text)
            if f == True:
                num += 1
    return ["6.1", num]


def etree_transform(evaluator, dst_style: str):
    trans_temp_var(evaluator)
    return evaluator


def program_transform(program_path, out_xml_path="./style/style.xml"):
    e = init_parse(program_path)
    trans_temp_var(e)
    save_file(doc, out_xml_path)


def xml_file_path(xml_path):
    global flag
    if not os.path.exists(transform_java_file):
        os.mkdir(transform_java_file)
    if not os.path.exists(save_xml_file):
        os.mkdir(save_xml_file)

    for xml_path_elem in xml_path:
        xmlfilepath = os.path.abspath(xml_path_elem)
        e = init_parse(xmlfilepath)
        flag = False
        trans_temp_var(e)
        if flag == True:
            str = xml_path_elem.split("\\")[-1]
            sub_dir = xml_path_elem.split("\\")[-2]
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file
