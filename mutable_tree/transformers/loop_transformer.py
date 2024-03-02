from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import ForToWhileVisitor, WhileToForVisitor


class LoopTransformer(CodeTransformer):
    name = "LoopTransformer"
    TRANSFORM_LOOP_FOR = "LoopTransformer.for_loop"
    TRANSFORM_LOOP_WHILE = "LoopTransformer.while_loop"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_LOOP_FOR, self.TRANSFORM_LOOP_WHILE]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_LOOP_FOR: WhileToForVisitor(),
            self.TRANSFORM_LOOP_WHILE: ForToWhileVisitor(),
        }[dst_style].visit(node)
