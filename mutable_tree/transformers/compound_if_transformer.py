from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import CompoundIfVisitor, NestedIfVisitor


class CompoundIfTransformer(CodeTransformer):
    name = "CompoundIfTransformer"
    TRANSFORM_IF_COMPOUND = "CompoundIfTransformer.if_compound"
    TRANSFORM_IF_NESTED = "CompoundIfTransformer.if_nested"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_IF_COMPOUND, self.TRANSFORM_IF_NESTED]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_IF_COMPOUND: CompoundIfVisitor(),
            self.TRANSFORM_IF_NESTED: NestedIfVisitor(),
        }[dst_style].visit(node)
