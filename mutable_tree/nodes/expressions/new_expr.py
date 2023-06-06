from ..node import Node, NodeType
from ..types import TypeIdentifier
from .expression import Expression
from .expression_list import ExpressionList
from ..utils import throw_invalid_type
from typing import List, Optional


class NewExpression(Expression):

    def __init__(self,
                 node_type: NodeType,
                 type: TypeIdentifier,
                 args: Optional[ExpressionList] = None):
        super().__init__(node_type)
        self.type = type
        self.args = args

    def _check_types(self):
        if self.node_type != NodeType.NEW_EXPR:
            throw_invalid_type(self.node_type, self)
        if self.type.node_type != NodeType.TYPE_IDENTIFIER:
            throw_invalid_type(self.type.node_type, self, 'type')
        if self.args is not None and self.args.node_type != NodeType.EXPRESSION_LIST:
            throw_invalid_type(self.args.node_type, self, 'args')

    def to_string(self) -> str:
        if self.args is not None:
            arg_list = ", ".join(arg.to_string() for arg in self.args.get_children())
            return f'new {self.type.to_string()}({arg_list})'
        else:
            return f'new {self.type.to_string()}'

    def get_children(self) -> List[Node]:
        if self.args is None:
            return [self.type, self.args]
        else:
            return [self.type]

    def get_children_names(self) -> List[str]:
        return ['type', 'args']
