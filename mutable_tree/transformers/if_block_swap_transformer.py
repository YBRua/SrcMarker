from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import NormalBlockSwapper, NegatedBlockSwapper


class IfBlockSwapTransformer(CodeTransformer):
    name = "IfBlockSwapTransformer"
    TRANSFORM_IF_BLOCK_NORMAL = "IfBlockSwapTransformer.normal"
    TRANSFORM_IF_BLOCK_NEGATED = "IfBlockSwapTransformer.negated"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_IF_BLOCK_NORMAL, self.TRANSFORM_IF_BLOCK_NEGATED]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_IF_BLOCK_NORMAL: NormalBlockSwapper(),
            self.TRANSFORM_IF_BLOCK_NEGATED: NegatedBlockSwapper(),
        }[dst_style].visit(node)
