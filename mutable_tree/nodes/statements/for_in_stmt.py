from ..node import Node, NodeType
from .statement import Statement
from .statement import is_statement
from .local_var_decl import DeclaratorType
from .declarators import Declarator, is_declarator
from ..expressions import Expression
from ..expressions import is_expression
from ..utils import throw_invalid_type
from typing import List


class ForInStatement(Statement):

    def __init__(self, node_type: NodeType, decl_type: DeclaratorType,
                 declarator: Declarator, iterable: Expression, body: Statement):
        super().__init__(node_type)
        self.decl_type = decl_type
        self.declarator = declarator
        self.iterable = iterable
        self.body = body
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FOR_IN_STMT:
            throw_invalid_type(self.node_type, self)
        if self.decl_type.node_type != NodeType.DECLARATOR_TYPE:
            throw_invalid_type(self.decl_type.node_type, self, attr='decl_type')
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr='declarator')
        if not is_expression(self.iterable):
            throw_invalid_type(self.iterable.node_type, self, attr='iterable')
        if not is_statement(self.body):
            throw_invalid_type(self.body.node_type, self, attr='body')

    def to_string(self) -> str:
        decl_type_str = self.decl_type.to_string()
        iter_str = self.declarator.to_string()
        iterable_str = self.iterable.to_string()
        body_str = self.body.to_string()

        return f'for ({decl_type_str} {iter_str} : {iterable_str})\n{body_str}'

    def get_children(self) -> List[Node]:
        return [self.decl_type, self.declarator, self.iterable, self.body]

    def get_children_names(self) -> List[str]:
        return ['decl_type', 'declarator', 'iterable', 'body']
