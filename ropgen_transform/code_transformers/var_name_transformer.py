from .transformer import CodeTransformer
from ropgen_transform.py import var_name_style
from ropgen_transform.xml_utils import init_parser, load_doc


class VarNameStyleTransformer(CodeTransformer):

    def __init__(self) -> None:
        super().__init__()

    def _is_camel_case(self, style_key: str):
        return style_key == '1.1'

    def _is_init_caps(self, style_key: str):
        return style_key == '1.2'

    def _is_underscore(self, style_key: str):
        return style_key == '1.3'

    def _is_init_underline(self, style_key: str):
        return style_key == '1.4'

    def get_available_transforms(self):
        return ['1.1', '1.2', '1.3', '1.4']

    def xml_transform(self, input_xml_path: str, src_style: str, dst_style: str,
                      output_xml_path: str):
        var_name_style.program_transform(input_xml_path, src_style, dst_style,
                                         output_xml_path)

    def xml_transform_all(self, input_xml_path: str, dst_style: str,
                          output_xml_path: str):
        var_name_style.program_transform_all(input_xml_path, dst_style, output_xml_path)

    def etree_transform_all(self, evaluator, dst_style: str):
        var_name_style.etree_transform(evaluator, dst_style)
        return evaluator

    def get_program_style(self, input_xml_path: str):
        xml_doc = load_doc(input_xml_path)
        xpath_evaluator = init_parser(xml_doc)
        camel_cases = []
        initcaps = []
        underscores = []
        init_underscores = []
        total_len = 0
        for decl in var_name_style.get_decls(xpath_evaluator):
            if len(var_name_style.get_declname(decl)) == 0:
                continue
            name_node = var_name_style.get_declname(decl)[0]
            name_text = name_node.text
            if name_text is None:
                name_node = var_name_style.get_declname(name_node)[0]
                name_text = name_node.text
            if var_name_style.is_camel_case(name_text):
                camel_cases.append(decl)
            elif var_name_style.is_initcap(name_text):
                initcaps.append(decl)
            elif var_name_style.is_init_underscore(name_text):
                init_underscores.append(decl)
            elif var_name_style.is_underscore(name_text):
                underscores.append(decl)
            if not var_name_style.is_all_lowercase(name_text) or '_' in name_text:
                total_len += 1
        return {
            '1.1': len(camel_cases),
            '1.2': len(initcaps),
            '1.3': len(underscores),
            '1.4': len(init_underscores),
        }
