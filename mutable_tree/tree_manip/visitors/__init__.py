# base
from .visitor import Visitor, TransformingVisitor, StatefulTransformingVisitor

# loop transformers
from .loop_while import ForToWhileVisitor
from .loop_for import WhileToForVisitor

# increment/decrement transformers
from .update_prefix import PrefixUpdateVisitor
from .update_postfix import PostfixUpdateVisitor
from .update_assign import AssignUpdateVisitor
from .update_binop import BinopUpdateVisitor

# condition transformers
from .ternary_to_if import TernaryToIfVisitor
from .switch_to_if import SwitchToIfVisitor
from .compound_if import CompoundIfVisitor
from .nested_if import NestedIfVisitor

# naming transformers
from .var_naming_style import (
    ToCamelCaseVisitor,
    ToPascalCaseVisitor,
    ToSnakeCaseVisitor,
    ToUnderscoreCaseVisitor,
)
from .identifier_rename import IdentifierRenamingVisitor, IdentifierAppendingVisitor

# variable decl transformers
from .var_same_type import SplitVarWithSameTypeVisitor, MergeVarWithSameTypeVisitor
from .var_pos import MoveVarDeclToHeadVisitor, MoveVarDeclToBeforeUsedVisitor
from .var_init import SplitVarInitAndDeclVisitor, MergeVarInitAndDeclVisitor

# block swap
from .block_swap import NormalBlockSwapper, NegatedBlockSwapper

# infinite loop condition
from .loop_condition import LoopLiteralOneVisitor, LoopLiteralTrueVisitor
