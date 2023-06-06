from .transformer import CodeTransformer
from ropgen_transform.py import init_declaration, var_init_split


class MultiDefinitionTransformer(CodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _is_var_decl_merged(self, style_key: str):
        return style_key == '8.1'

    def _is_var_decl_separated(self, style_key: str):
        return style_key == '8.2'

    def get_available_transforms(self):
        return ['8.1', '8.2']

    def etree_transform_all(self, evaluator, dst_style: str):
        if self._is_var_decl_merged(dst_style):
            init_declaration.transform(evaluator)
        elif self._is_var_decl_separated(dst_style):
            var_init_split.transform_standalone_stmts(evaluator)
        else:
            raise ValueError(f'Invalid dst_style: {dst_style}')
        return evaluator

    def xml_transform_all(self, input_xml_path: str, dst_style: str,
                          output_xml_path: str):
        self.xml_transform(input_xml_path, None, dst_style, output_xml_path)

    def xml_transform(self, input_xml_path: str, src_style: str, dst_style: str,
                      output_xml_path: str):
        if self._is_var_decl_merged(dst_style):
            init_declaration.program_transform(input_xml_path, output_xml_path)
        elif self._is_var_decl_separated(dst_style):
            var_init_split.program_transform(input_xml_path, output_xml_path)
        else:
            raise ValueError(f'Invalid dst_style: {dst_style}')
