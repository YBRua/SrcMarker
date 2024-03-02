from enum import Enum
from ..node import Node, NodeType
from .statement import Statement
from .statement import is_statement
from .local_var_decl import DeclaratorType
from .declarators import Declarator, is_declarator
from ..expressions import Expression
from ..expressions import is_expression
from ..utils import throw_invalid_type
from typing import List


class ForInType(Enum):
    COLON = ":"
    IN = "in"
    OF = "of"


_forin_type_map = {
    ":": ForInType.COLON,
    "in": ForInType.IN,
    "of": ForInType.OF,
}


def get_forin_type(op: str) -> ForInType:
    return _forin_type_map[op]


class ForInStatement(Statement):
    def __init__(
        self,
        node_type: NodeType,
        decl_type: DeclaratorType,
        declarator: Declarator,
        iterable: Expression,
        body: Statement,
        forin_type: ForInType,
        is_async: bool = False,
    ):
        super().__init__(node_type)
        self.decl_type = decl_type
        self.declarator = declarator
        self.iterable = iterable
        self.body = body
        self.forin_type = forin_type
        self.is_async = is_async
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FOR_IN_STMT:
            throw_invalid_type(self.node_type, self)
        if self.decl_type.node_type != NodeType.DECLARATOR_TYPE:
            throw_invalid_type(self.decl_type.node_type, self, attr="decl_type")
        if not is_declarator(self.declarator) and not is_expression(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr="declarator")
        if not is_expression(self.iterable):
            throw_invalid_type(self.iterable.node_type, self, attr="iterable")
        if not is_statement(self.body):
            throw_invalid_type(self.body.node_type, self, attr="body")

    def get_children(self) -> List[Node]:
        return [self.decl_type, self.declarator, self.iterable, self.body]

    def get_children_names(self) -> List[str]:
        return ["decl_type", "declarator", "iterable", "body"]
