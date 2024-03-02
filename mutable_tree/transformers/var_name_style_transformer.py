from mutable_tree.nodes import Node
from .code_transformer import CodeTransformer
from ..tree_manip.visitors import (
    ToCamelCaseVisitor,
    ToPascalCaseVisitor,
    ToSnakeCaseVisitor,
    ToUnderscoreCaseVisitor,
)


class VarNameStyleTransformer(CodeTransformer):
    name = "VarNameStyleTransformer"
    TRANSFORM_CAMEL_CASE = "VarNameStyleTransformer.camel_case"
    TRANSFORM_PASCAL_CASE = "VarNameStyleTransformer.pascal_case"
    TRANSFORM_SNAKE_CASE = "VarNameStyleTransformer.snake_case"
    TRANSFORM_UNDERSCORE_CASE = "VarNameStyleTransformer.underscore_case"

    def __init__(self) -> None:
        super().__init__()

    def get_available_transforms(self):
        return [
            self.TRANSFORM_CAMEL_CASE,
            self.TRANSFORM_PASCAL_CASE,
            self.TRANSFORM_SNAKE_CASE,
            self.TRANSFORM_UNDERSCORE_CASE,
        ]

    def mutable_tree_transform(self, node: Node, dst_style: str):
        return {
            self.TRANSFORM_CAMEL_CASE: ToCamelCaseVisitor(),
            self.TRANSFORM_PASCAL_CASE: ToPascalCaseVisitor(),
            self.TRANSFORM_SNAKE_CASE: ToSnakeCaseVisitor(),
            self.TRANSFORM_UNDERSCORE_CASE: ToUnderscoreCaseVisitor(),
        }[dst_style].visit(node)
