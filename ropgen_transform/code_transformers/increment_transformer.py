from .transformer import CodeTransformer
from ..xml_utils import load_doc, init_parser
from ropgen_transform.py import incr_opr_prepost, incr_opr_usage


class IncrementTransformer(CodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return ['10.1', '10.2', '10.3', '10.4']

    def xml_transform(self, input_xml_path: str, src_style: str, dst_style: str,
                      output_xml_path: str):
        incr_opr_prepost.program_transform(input_xml_path, src_style, dst_style,
                                           output_xml_path)

    def xml_transform_all(self, input_xml_path: str, dst_style: str,
                          output_xml_path: str):
        incr_opr_prepost.program_transform_all(input_xml_path, dst_style, output_xml_path)

    def etree_transform_all(self, evaluator, dst_style: str):
        incr_opr_prepost.etree_transform(evaluator, dst_style)
        return evaluator

    def get_program_style(self, input_xml_path: str):
        evaluator = init_parser(load_doc(input_xml_path))
        prefix_incrs_len = 0
        postfix_incrs_len = 0
        incr_plus_literals_len = 0
        incr_full_len = 0
        for expr, style in incr_opr_usage.get_incr_exprs(evaluator, 3):
            if style == 1:
                postfix_incrs_len += 1
            elif style == 2:
                prefix_incrs_len += 1
            elif style == 3:
                incr_full_len += 1
            elif style == 4:
                incr_plus_literals_len += 1

        return {
            '10.1': postfix_incrs_len,
            '10.2': prefix_incrs_len,
            '10.3': incr_full_len,
            '10.4': incr_plus_literals_len
        }
