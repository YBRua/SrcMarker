"""

Local variable definition location

"""

import os
import sys

from lxml import etree

doc = None
flag = False
ns = {
    'src': 'http://www.srcML.org/srcML/src',
    'cpp': 'http://www.srcML.org/srcML/cpp',
    'pos': 'http://www.srcML.org/srcML/position'
}

save_xml_file = './transform_xml_file/temp_var_head'
transform_java_file = './target_author_file/transform_java/temp_var_head'


def get_block_cons(e):
    return e('//src:block_content')


def init_parse(e):
    global doc
    doc = etree.parse(e)
    e = etree.XPathEvaluator(doc, namespaces=ns)
    return e


def get_decls(block_con):
    return block_con.xpath('src:decl_stmt', namespaces=ns)


def get_allnames(decl):
    return decl.xpath('.//src:name', namespaces=ns)


def get_instances(e):
    instances = []
    block_cons = get_block_cons(e)
    for block_con in block_cons:
        vars_names = []
        decls = get_decls(block_con)
        if len(decls) == 0: continue
        decl_index = 0
        for decl_stmt in block_con:
            if decl_stmt.tag != '{http://www.srcML.org/srcML/src}decl_stmt':
                break
            for dec in decl_stmt:
                if len(dec[1]) == 0:
                    vars_names.append("".join(dec[1].itertext()))
                else:
                    vars_names.append("".join(dec[1][0].itertext()))
            decl_index += 1
        index = decl_index
        #print(decl_index)

        if len(decls) <= decl_index:
            continue
        #Start from the previous one of decls[decl_index] and move all the ones below
        prev = decls[decl_index]

        for decl in decls[decl_index:]:
            #   print(decl.tag)
            rep_var_flag = False
            curr_names = get_allnames(decl)
            decl_stmt_index = block_con.index(decl)
            for decl_stmt in block_con[0:decl_stmt_index]:
                decl_var_names = [
                    name.text for name in get_allnames(decl_stmt) if len(name) == 0
                ]
                vars_names += decl_var_names
            for curr_name in curr_names:
                if len(curr_name) == 0:
                    currname_var = ''.join(curr_name.itertext())
                    #print(currname_var)
                    if currname_var in vars_names:
                        rep_var_flag = True

                        for d in decl:
                            if d.tag == '{http://www.srcML.org/srcML/src}comment':
                                continue
                            if len(d) <= 1: continue
                            if len(d[1]) == 0:
                                vars_names.append("".join(d[1].itertext()))
                            else:
                                vars_names.append("".join(d[1][0].itertext()))
                        break
            if rep_var_flag == True: continue

            for d in decl:
                if d.tag == '{http://www.srcML.org/srcML/src}comment': continue
                if len(d[1]) == 0:
                    vars_names.append("".join(d[1].itertext()))
                else:
                    vars_names.append("".join(d[1][0].itertext()))
            instances.append((prev, index, decl, block_con))
            index += 1
    return instances


def trans_temp_var(e, ignore_list=[], instances=None):
    global flag
    flag = False
    # Only consider the temporary variables in the loop and the condition body Get the <block_content> tag
    decls = [
        get_instances(e) if instances is None else
        (instance[0] for instance in instances if len(instance) > 0)
    ]
    # Get all initial temporary variables in the statement block

    tree_root = e('/*')[0].getroottree()
    new_ignore_list = []

    for item in decls:
        for inst_tuple in item:
            prev = inst_tuple[0]
            decl_index = inst_tuple[1]
            decl = inst_tuple[2]

            if (decl.getparent()[0].tag == '{http://www.srcML.org/srcML/src}case'):
                continue
            else:
                context = inst_tuple[3]

            prev_path = tree_root.getpath(prev)
            if prev_path in ignore_list:
                continue
            context.insert(decl_index, decl)
            flag = True

            new_ignore_list.append(prev_path)

    return flag, tree_root, new_ignore_list


def save_file(doc, param):
    with open(param, 'w') as f:
        f.write(etree.tostring(doc).decode('utf-8'))


def get_style(xmlfilepath):

    e = init_parse(xmlfilepath)
    num = 0
    block_cons = get_block_cons(e)

    for block_con in block_cons:
        vars_names = []
        decls = get_decls(block_con)
        decl_index = 0
        for decl_stmt in block_con:
            if decl_stmt.tag != '{http://www.srcML.org/srcML/src}decl_stmt':
                break
            for dec in decl_stmt:
                vars_names.append("".join(dec[1].itertext()))
            decl_index += 1
        index = decl_index
        for decl in decls[decl_index:]:

            rep_var_flag = False
            curr_names = get_allnames(decl)
            # print(vars_names)
            for curr_name in curr_names:
                if len(curr_name) == 0:
                    currname_var = ''.join(curr_name.itertext())
                    # print(currname_var)
                    if currname_var in vars_names:
                        rep_var_flag = True

                        for d in decl:
                            if d.tag == '{http://www.srcML.org/srcML/src}comment':
                                continue
                            vars_names.append("".join(d[1].itertext()))
                        break
            if rep_var_flag == True: continue
            for d in decl:
                if d.tag == '{http://www.srcML.org/srcML/src}comment': continue
                vars_names.append("".join(d[1].itertext()))
            block_con.insert(index, decl)
            num += 1
            index += 1
    return ['6.2', num]


def program_transform(program_path, out_xml_path='./style/style.xml'):
    e = init_parse(program_path)
    trans_temp_var(e)
    save_file(doc, out_xml_path)


def etree_transform(evaluator, dst_style: str):
    trans_temp_var(evaluator)
    return evaluator


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
            str = xml_path_elem.split('\\')[-1]
            sub_dir = xml_path_elem.split('\\')[-2]
            if not os.path.exists(os.path.join(save_xml_file, sub_dir)):
                os.mkdir(os.path.join(save_xml_file, sub_dir))
            save_file(doc, os.path.join(save_xml_file, sub_dir, str))
    return save_xml_file, transform_java_file
