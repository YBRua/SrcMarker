from .transformer import RopGenCodeTransformer
from ..xml_utils import load_doc, init_parser
from ropgen_transform.py import var_init_merge, var_init_pos


class LocalVarInitTransformer(RopGenCodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _is_init_merged(self, style_key: str):
        return style_key == '7.1'

    def _is_init_separated(self, style_key: str):
        return style_key == '7.2'

    def get_available_transforms(self):
        return ['7.1', '7.2']

    def etree_transform_all(self, evaluator, dst_style: str):
        if self._is_init_merged(dst_style):
            var_init_merge.etree_transform(evaluator, dst_style)
        elif self._is_init_separated(dst_style):
            var_init_pos.etree_transform(evaluator, dst_style)
        else:
            raise ValueError(f'Invalid dst_style: {dst_style}')
        return evaluator

    def xml_transform_all(self, input_xml_path: str, dst_style: str,
                          output_xml_path: str):
        self.xml_transform(input_xml_path, None, dst_style, output_xml_path)

    def xml_transform(self, input_xml_path: str, src_style: str, dst_style: str,
                      output_xml_path: str):
        if self._is_init_merged(dst_style):
            var_init_merge.program_transform(input_xml_path, output_xml_path)
        elif self._is_init_separated(dst_style):
            var_init_pos.program_transform(input_xml_path, output_xml_path)
        else:
            raise ValueError(f'Invalid dst_style: {dst_style}')

    def get_program_style(self, input_xml_path: str):
        evaluator = init_parser(load_doc(input_xml_path))
        separated_inits = var_init_merge.get_separate_inits(evaluator)
        merged_inits = var_init_pos.get_decl_init_stmts(evaluator)
        separate_inits_len = len(separated_inits)
        merged_inits_len = len(merged_inits)

        return {'7.1': merged_inits_len, '7.2': separate_inits_len}
