from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import SplitVarInitAndDeclVisitor, MergeVarInitAndDeclVisitor


class VarInitTransformer(CodeTransformer):
    name = "VarInitTransformer"
    TRANSFORM_INIT_SPLIT = "VarInitTransformer.split"
    TRANSFORM_INIT_MERGE = "VarInitTransformer.merge"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [
            self.TRANSFORM_INIT_SPLIT,
            self.TRANSFORM_INIT_MERGE,
        ]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_INIT_SPLIT: SplitVarInitAndDeclVisitor(),
            self.TRANSFORM_INIT_MERGE: MergeVarInitAndDeclVisitor(),
        }[dst_style].visit(node)
