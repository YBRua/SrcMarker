from .transformer import RopGenCodeTransformer
from .var_name_transformer import VarNameStyleTransformer
from .local_var_def_transformer import LocalVarDefTransformer
from .local_var_init_transformer import LocalVarInitTransformer
from .increment_transformer import IncrementTransformer
from .loop_transformer import LoopTransformer
from .condition_transformer import ConditionTransformer
from .compound_if_transformer import CompoundIfTransformer
from .multi_definition_transformer import MultiDefinitionTransformer
from .incr_assign_transformer import IncrAssignTransformer
from typing import List


def get_transformers() -> List[RopGenCodeTransformer]:
    return [
        VarNameStyleTransformer(),
        LocalVarDefTransformer(),
        LocalVarInitTransformer(),
        IncrementTransformer(),
        LoopTransformer(),
        ConditionTransformer(),
        CompoundIfTransformer(),
        MultiDefinitionTransformer(),
        IncrAssignTransformer(),
    ]
