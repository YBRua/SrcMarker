from .transformer import RopGenCodeTransformer
from ropgen_transform.py import re_temp, temporary_var


class LocalVarDefTransformer(RopGenCodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _is_defined_beginning(self, style_key: str):
        return style_key == '6.1'

    def _is_defined_first_use(self, style_key: str):
        return style_key == '6.2'

    def get_available_transforms(self):
        return ['6.1', '6.2']

    def etree_transform_all(self, evaluator, dst_style: str):
        if self._is_defined_beginning(dst_style):
            temporary_var.etree_transform(evaluator, dst_style)
        elif self._is_defined_first_use(dst_style):
            re_temp.etree_transform(evaluator, dst_style)
        else:
            raise ValueError(f'Invalid dst_style: {dst_style}')
        return evaluator

    def xml_transform_all(self, input_xml_path: str, dst_style: str,
                          output_xml_path: str):
        self.xml_transform(input_xml_path, None, dst_style, output_xml_path)

    def xml_transform(self, input_xml_path: str, src_style: str, dst_style: str,
                      output_xml_path: str):
        if self._is_defined_beginning(dst_style):
            temporary_var.program_transform(input_xml_path, output_xml_path)
        elif self._is_defined_first_use(dst_style):
            re_temp.program_transform(input_xml_path, output_xml_path)
        else:
            raise ValueError(f'Invalid dst_style: {dst_style}')

    def get_program_style(self, input_xml_path: str):
        l1 = temporary_var.get_style(input_xml_path)
        l2 = re_temp.get_style(input_xml_path)

        return {'6.1': l1[1], '6.2': l2[1]}
