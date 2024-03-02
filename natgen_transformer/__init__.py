from .transformation_base import NatGenBaseTransformer

from .for_while_transformation import ForWhileTransformer
from .block_swap_transformation import BlockSwapTransformer
from .confusion_remove_transformation import ConfusionRemoveTransformer
from .dead_code_insersion import DeadCodeInserter
from .operand_swap_transformation import OperandSwapTransformer
from .var_renaming_transformation import VarRenamingTransformer
from typing import List


def get_natgen_transformers(
    parser_path: str, language: str
) -> List[NatGenBaseTransformer]:
    return [
        ForWhileTransformer(parser_path, language),
        BlockSwapTransformer(parser_path, language),
        ConfusionRemoveTransformer(parser_path, language),
        DeadCodeInserter(parser_path, language),
        OperandSwapTransformer(parser_path, language),
        VarRenamingTransformer(parser_path, language),
    ]
