from .style_stats import calculate_proportion, get_dominating_styles
from .code_transformers import (
    RopGenCodeTransformer,
    VarNameStyleTransformer,
    LoopTransformer,
    IncrementTransformer,
    LocalVarDefTransformer,
    LocalVarInitTransformer,
    CompoundIfTransformer,
    ConditionTransformer,
    MultiDefinitionTransformer,
    IncrAssignTransformer,
)

TRANFORM_ID_TO_TRANSFORMER = {
    1: VarNameStyleTransformer(),
    6: LocalVarDefTransformer(),
    7: LocalVarInitTransformer(),
    8: MultiDefinitionTransformer(),
    9: IncrAssignTransformer(),
    10: IncrementTransformer(),
    20: LoopTransformer(),
    21: ConditionTransformer(),
    22: CompoundIfTransformer(),
}


def get_transformer(transform_type: int) -> RopGenCodeTransformer:
    return TRANFORM_ID_TO_TRANSFORMER[transform_type]
