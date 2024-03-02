from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import LoopLiteralOneVisitor, LoopLiteralTrueVisitor


class InfiniteLoopTransformer(CodeTransformer):
    name = "InfiniteLoopTransformer"
    TRANSFORM_INFLOOP_TRUE = "InfiniteLoopTransformer.literal_true"
    TRANSFORM_INFLOOP_ONE = "InfiniteLoopTransformer.literal_1"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_INFLOOP_TRUE, self.TRANSFORM_INFLOOP_ONE]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_INFLOOP_TRUE: LoopLiteralOneVisitor(),
            self.TRANSFORM_INFLOOP_ONE: LoopLiteralTrueVisitor(),
        }[dst_style].visit(node)
