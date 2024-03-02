from .transformer import RopGenCodeTransformer
from ropgen_transform.py import assign_combine, assign_value


class IncrAssignTransformer(RopGenCodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _is_incr_assign_separated(self, style_key: str):
        return style_key == "9.1"

    def _is_incr_assign_combined(self, style_key: str):
        return style_key == "9.2"

    def get_available_transforms(self):
        return ["9.1", "9.2"]

    def etree_transform_all(self, evaluator, dst_style: str):
        if self._is_incr_assign_separated(dst_style):
            assign_value.trans_tree(evaluator)
        elif self._is_incr_assign_combined(dst_style):
            assign_combine.trans_tree(evaluator)
        else:
            raise ValueError(f"Invalid dst_style: {dst_style}")
        return evaluator

    def xml_transform_all(
        self, input_xml_path: str, dst_style: str, output_xml_path: str
    ):
        self.xml_transform(input_xml_path, None, dst_style, output_xml_path)

    def xml_transform(
        self, input_xml_path: str, src_style: str, dst_style: str, output_xml_path: str
    ):
        if self._is_incr_assign_separated(dst_style):
            assign_value.program_transform(input_xml_path, output_xml_path)
        elif self._is_incr_assign_combined(dst_style):
            assign_combine.program_transform(input_xml_path, output_xml_path)
        else:
            raise ValueError(f"Invalid dst_style: {dst_style}")
