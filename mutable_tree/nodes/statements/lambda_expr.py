from ..node import Node, NodeType
from ..expressions import Expression
from ..expressions import is_expression
from .block_stmt import BlockStatement
from .func_declaration import FormalParameterList
from ..miscs import ModifierList
from ..utils import throw_invalid_type
from typing import List, Union, Optional


class LambdaExpression(Expression):
    def __init__(
        self,
        node_type: NodeType,
        lambda_params: FormalParameterList,
        body: Union[Expression, BlockStatement],
        parenthesized: bool = False,
        modifiers: Optional[ModifierList] = None,
    ):
        super().__init__(node_type)
        self.params = lambda_params
        self.body = body
        self.parenthesized = parenthesized
        self.modifiers = modifiers
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.LAMBDA_EXPR:
            throw_invalid_type(self.node_type, self)
        if self.params.node_type != NodeType.FORMAL_PARAMETER_LIST:
            throw_invalid_type(self.params.node_type, self, "params")
        if self.body.node_type != NodeType.BLOCK_STMT and not is_expression(self.body):
            throw_invalid_type(self.body.node_type, self, "body")
        if (
            self.modifiers is not None
            and self.modifiers.node_type != NodeType.MODIFIER_LIST
        ):
            throw_invalid_type(self.modifiers.node_type, self, "modifiers")

    def get_children(self) -> List[Node]:
        return [self.params, self.body, self.modifiers]

    def get_children_names(self) -> List[str]:
        return ["params", "body", "modifiers"]
