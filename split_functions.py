import os
from tqdm import tqdm
from lxml import etree
from ropgen_transform import srcml_interface
from ropgen_transform.xml_utils import load_doc, init_parser


def get_functions(evaluator):
    return evaluator("//src:function")


if __name__ == "__main__":
    DATASET_DIR = "data/github_java"
    OUTPUT_DIR = "data/github_java_funcs"
    LANG = "java"

    XML_HEADER = r'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    XML_FOOTER = r"</unit>"

    tot_funcs = 0
    prog = tqdm(os.listdir(DATASET_DIR))
    for author in prog:
        prog.set_description(f"AUTHOR: {author:<25}")
        author_src_dir = os.path.join(DATASET_DIR, author)
        author_dst_dir = os.path.join(OUTPUT_DIR, author)
        if not os.path.exists(author_dst_dir):
            os.mkdir(author_dst_dir)

        for file in os.listdir(author_src_dir):
            file_path = os.path.join(author_src_dir, file)
            file_path_noext, ext = os.path.splitext(file)
            file_xml = os.path.join(author_dst_dir, f"{file_path_noext}.xml")

            srcml_interface.srcml_program_to_xml(file_path, file_xml)
            doc = load_doc(file_xml)
            evaluator = init_parser(doc)
            funcs = get_functions(evaluator)

            for i, func in enumerate(funcs):
                tot_funcs += 1
                func_xml_output_path = os.path.join(
                    author_dst_dir, f"{file_path_noext}.{i}.xml"
                )
                func_file_output_path = os.path.join(
                    author_dst_dir, f"{file_path_noext}.{i}.{LANG}"
                )

                xml_premble = (
                    r'<unit xmlns="http://www.srcML.org/srcML/src" '
                    #    r'xmlns:cpp="http://www.srcML.org/srcML/cpp" '
                    r'xmlns:pos="http://www.srcML.org/srcML/position" '
                    r'revision="1.0.0" language="C" '
                    rf'filename="{func_file_output_path}" '
                    r'pos:tabs="8">'
                )

                with open(func_xml_output_path, "w") as fo:
                    fo.write(XML_HEADER + "\n")
                    fo.write(xml_premble + "\n")
                    estr = etree.tostring(func).decode("utf8")
                    estr = estr[: estr.rfind(">") + 1]
                    fo.write(estr)
                    fo.write(XML_FOOTER + "\n")

                srcml_interface.srcml_xml_to_program(
                    func_xml_output_path, func_file_output_path
                )

                os.remove(func_xml_output_path)
            os.remove(file_xml)

    print(f"Total functions: {tot_funcs}")
