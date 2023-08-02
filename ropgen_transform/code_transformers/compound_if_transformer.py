from .transformer import RopGenCodeTransformer
from ..xml_utils import load_doc, init_parser
from ropgen_transform.py import if_spilt, if_combine


class CompoundIfTransformer(RopGenCodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _is_combined_if(self, style_key: str):
        return style_key == '22.1'

    def _is_nested_if(self, style_key: str):
        return style_key == '22.2'

    def get_available_transforms(self):
        return ['22.1', '22.2']

    def etree_transform_all(self, evaluator, dst_style: str):
        if self._is_combined_if(dst_style):
            if_combine.etree_transform(evaluator, dst_style)
        elif self._is_nested_if(dst_style):
            if_spilt.etree_transform(evaluator, dst_style)
        else:
            raise ValueError(f'Invalid dst_style: {dst_style}')
        return evaluator

    def xml_transform(self, input_xml_path: str, src_style: str, dst_style: str,
                      output_xml_path: str):
        if self._is_combined_if(dst_style):
            if_combine.program_transform(input_xml_path, output_xml_path)
        elif self._is_nested_if(dst_style):
            if_spilt.program_transform(input_xml_path, output_xml_path)
        else:
            raise ValueError(f'Invalid dst_style: {dst_style}')

    def xml_transform_all(self, input_xml_path: str, dst_style: str, output_xml_path: str):
        self.xml_transform(input_xml_path, None, dst_style, output_xml_path)

    def get_program_style(self, input_xml_path: str):
        evaluator = init_parser(load_doc(input_xml_path))
        n_if_split = if_spilt.count(evaluator)
        n_if_combine = if_combine.count(evaluator)

        return {'22.1': n_if_combine, '22.2': n_if_split}
