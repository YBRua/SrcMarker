from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import (
    MoveVarDeclToHeadVisitor,
    MoveVarDeclToBeforeUsedVisitor,
)


class VarDeclLocationTransformer(CodeTransformer):
    name = "VarDeclLocationTransformer"
    TRANSFORM_VARDECL_BLOCK_START = "VarDeclLocationTransformer.block_start"
    TRANSFORM_VARDECL_FIRST_USE = "VarDeclLocationTransformer.first_use"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [
            self.TRANSFORM_VARDECL_BLOCK_START,
            self.TRANSFORM_VARDECL_FIRST_USE,
        ]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_VARDECL_BLOCK_START: MoveVarDeclToHeadVisitor(),
            self.TRANSFORM_VARDECL_FIRST_USE: MoveVarDeclToBeforeUsedVisitor(),
        }[dst_style].visit(node)
