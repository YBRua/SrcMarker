from .transformer import RopGenCodeTransformer
from ..xml_utils import load_doc, init_parser
from ropgen_transform.py import switch_if, ternary


class ConditionTransformer(RopGenCodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _is_switch_if(self, style_key: str):
        return style_key == "21.1"

    def _is_ternary(self, style_key: str):
        return style_key == "21.2"

    def get_available_transforms(self):
        return ["21.1", "21.2"]

    def etree_transform_all(self, evaluator, dst_style: str):
        if self._is_switch_if(dst_style):
            switch_if.etree_transform(evaluator, dst_style)
        elif self._is_ternary(dst_style):
            ternary.etree_transform(evaluator, dst_style)
        else:
            raise ValueError(f"Invalid dst_style: {dst_style}")
        return evaluator

    def xml_transform(
        self, input_xml_path: str, src_style: str, dst_style: str, output_xml_path: str
    ):
        if self._is_switch_if(dst_style):
            switch_if.program_transform(input_xml_path, output_xml_path)
        elif self._is_ternary(dst_style):
            ternary.program_transform(input_xml_path, output_xml_path)
        else:
            raise ValueError(f"Invalid dst_style: {dst_style}")

    def xml_transform_all(
        self, input_xml_path: str, dst_style: str, output_xml_path: str
    ):
        self.xml_transform(input_xml_path, None, dst_style, output_xml_path)

    def get_program_style(self, input_xml_path: str):
        evaluator = init_parser(load_doc(input_xml_path))
        n_switch_if = switch_if.count(evaluator)
        n_ternary = ternary.count(evaluator)
        return {"21.1": n_switch_if, "21.2": n_ternary}
