from .code_transformer import CodeTransformer
from .compound_if_transformer import CompoundIfTransformer
from .condition_transformer import ConditionTransformer
from .loop_transformer import LoopTransformer
from .update_transformer import UpdateTransformer
from .same_type_decl_transformer import SameTypeDeclarationTransformer
from .var_name_style_transformer import VarNameStyleTransformer
from .if_block_swap_transformer import IfBlockSwapTransformer
from .var_init_transformer import VarInitTransformer
from .var_decl_pos_transformer import VarDeclLocationTransformer
from .infinite_loop_transformer import InfiniteLoopTransformer
from typing import List


def get_all_transformers() -> List[CodeTransformer]:
    return [
        CompoundIfTransformer(),
        ConditionTransformer(),
        LoopTransformer(),
        UpdateTransformer(),
        SameTypeDeclarationTransformer(),
        VarNameStyleTransformer(),
        IfBlockSwapTransformer(),
        VarInitTransformer(),
        VarDeclLocationTransformer(),
        InfiniteLoopTransformer(),
    ]
