from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import (
    SplitVarWithSameTypeVisitor,
    MergeVarWithSameTypeVisitor,
)


class SameTypeDeclarationTransformer(CodeTransformer):
    name = "SameTypeDeclarationTransformer"
    TRANSFORM_SAME_TYPE_SPLIT = "SameTypeDeclarationTransformer.split"
    TRANSFORM_SAME_TYPE_MERGE = "SameTypeDeclarationTransformer.merge"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [self.TRANSFORM_SAME_TYPE_SPLIT, self.TRANSFORM_SAME_TYPE_MERGE]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_SAME_TYPE_SPLIT: SplitVarWithSameTypeVisitor(),
            self.TRANSFORM_SAME_TYPE_MERGE: MergeVarWithSameTypeVisitor(),
        }[dst_style].visit(node)
