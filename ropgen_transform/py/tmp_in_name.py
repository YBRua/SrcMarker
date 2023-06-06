import os

from lxml import etree

doc = None
ns = {
    'src': 'http://www.srcML.org/srcML/src',
    'cpp': 'http://www.srcML.org/srcML/cpp',
    'pos': 'http://www.srcML.org/srcML/position'
}


def init_parser(file):
    global doc
    doc = etree.parse(file)
    e = etree.XPathEvaluator(doc, namespaces=ns)
    return e


def save_tree_to_file(tree, file):
    with open(file, 'w') as f:
        f.write(etree.tostring(tree).decode('utf8'))


def get_function(e):
    return e('.//src:function')


def get_fors(elem):
    return elem.xpath('.//src:for', namespaces=ns)


def alter_for_name(for_elems, author_for_names, typedef_names, func_s):
    author_list_elems = []
    for author_name in author_for_names:
        author_name_elems = author_name.xpath('src:init/src:decl/src:name', namespaces=ns)
        author_name_elem = author_name_elems[0] if len(author_name_elems) > 0 else (
            author_name.xpath('src:init//src:argument_list//src:name', namespaces=ns)[0]
            if
            len(author_name.xpath('src:init//src:argument_list//src:name',
                                  namespaces=ns)) > 0 else None)
        if author_name_elem is None: continue
        if author_name_elem.text is None: continue
        author_list_elems.append(author_name_elem.text)
    author_list_elems.sort(key=lambda i: len(i), reverse=True)
    author_list_elems = sorted(set(author_list_elems), key=author_list_elems.index)
    for for_elem in for_elems:
        if len(author_list_elems) == 0: break
        decl = for_elem.xpath('src:control/src:init/src:decl/src:name', namespaces=ns)
        #decl = decl[0] if len(decl) != 0 else for_elem.xpath('src:control/src:init//src:argument_list//src:name',namespaces=ns)[0] if len(for_elem.xpath('src:control/src:init//src:argument_list//src:name',namespaces=ns)) > 0 else None
        if len(decl) == 0: continue
        decl = decl[0]
        if decl == None: continue
        pre_name = decl.text
        later_name = author_list_elems[0]
        name_elems = for_elem.xpath('.//src:name', namespaces=ns)
        name_list = []
        for name_elem in name_elems:
            name_list.append(name_elem.text)
        if later_name in name_list:
            continue
        #
        if later_name in typedef_names:
            continue
        #
        class_name = func_s.xpath(
            './src:template/src:parameter_list/src:parameter/src:name', namespaces=ns)
        if class_name is not None and len(class_name) == 1:
            class_name = class_name[0]
            if class_name.text == later_name: continue
        for name_elem in name_elems:
            if name_elem.text == pre_name:
                name_elem.text = later_name
        del author_list_elems[0]


def get_author_for_names(dst_author):
    author_names = []
    file_list = os.listdir(dst_author) if os.path.isdir(dst_author) else [dst_author]
    for file_elem in file_list:
        if not file_elem.endswith('.xml'): continue
        file_elem = os.path.join(dst_author if os.path.isdir(dst_author) else '',
                                 file_elem)
        e = init_parser(file_elem)
        for elem in e('.//src:for/src:control'):
            author_names.append(elem)
    return author_names


def transfrom_forName(src_author, dst_author):
    # author
    author_for_names = get_author_for_names(dst_author)
    #
    e = init_parser(src_author)

    func_elems = get_function(e)
    #
    typedef_names = []
    for name_elem in e('//src:typedef/src:name'):
        typedef_names.append(name_elem.text)
    #print(typedef_names)

    for func_s in func_elems:
        func_ss = func_s.xpath('./src:block/src:block_content', namespaces=ns)
        if len(func_ss) == 0: continue
        func_elem = func_ss[0]
        # 111_for
        for_elems = get_fors(func_elem)
        alter_for_name(for_elems, author_for_names, typedef_names, func_s)


def trans_tree(src_author, dst_author):
    #for
    transfrom_forName(src_author, dst_author)
    save_tree_to_file(doc, './style/style.xml')


if __name__ == '__main__':
    src_author = '../demo1.xml'
    dst_author = '../xml_file/xyiyy'
    trans_tree(src_author, dst_author)
    save_tree_to_file(doc, '../demo2.xml')