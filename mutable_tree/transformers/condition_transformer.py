from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import SwitchToIfVisitor, TernaryToIfVisitor


class ConditionTransformer(CodeTransformer):
    name = "ConditionTransformer"
    TRANSFORM_COND_SWITCH = "ConditionTransformer.switch"
    TRANSFORM_COND_TERNARY = "ConditionTransformer.ternary"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_COND_SWITCH, self.TRANSFORM_COND_TERNARY]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_COND_SWITCH: SwitchToIfVisitor(),
            self.TRANSFORM_COND_TERNARY: TernaryToIfVisitor(),
        }[dst_style].visit(node)
