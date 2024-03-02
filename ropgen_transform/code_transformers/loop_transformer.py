from .transformer import RopGenCodeTransformer
from ..xml_utils import init_parser, load_doc
from ropgen_transform.py import for_while, while_for


class LoopTransformer(RopGenCodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _is_for_loop(self, style_key: str):
        return style_key == "20.1"

    def _is_while_loop(self, style_key: str):
        return style_key == "20.2"

    def get_available_transforms(self):
        return ["20.1", "20.2"]

    def xml_transform(
        self, input_xml_path: str, src_style: str, dst_style: str, output_xml_path: str
    ):
        if self._is_for_loop(dst_style):
            while_for.program_transform(input_xml_path, output_xml_path)
        elif self._is_while_loop(dst_style):
            for_while.program_transform(input_xml_path, output_xml_path)
        else:
            raise ValueError(f"Invalid dst_style: {dst_style}")

    def xml_transform_all(
        self, input_xml_path: str, dst_style: str, output_xml_path: str
    ):
        self.xml_transform(input_xml_path, None, dst_style, output_xml_path)

    def etree_transform_all(self, evaluator, dst_style: str):
        if self._is_for_loop(dst_style):
            while_for.etree_transform(evaluator)
        elif self._is_while_loop(dst_style):
            for_while.etree_transform(evaluator)
        else:
            raise ValueError(f"Invalid dst_style: {dst_style}")

        return evaluator

    def get_program_style(self, input_xml_path: str):
        evaluator = init_parser(load_doc(input_xml_path))
        n_fors = for_while.count(evaluator)
        n_whiles = while_for.count(evaluator)

        return {"20.1": n_fors, "20.2": n_whiles}
