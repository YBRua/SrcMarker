from ..node import Node, NodeType
from .statement import Statement
from .statement import is_statement
from ..expressions import Expression
from ..expressions import is_expression
from ..utils import throw_invalid_type
from typing import List


class WithStatement(Statement):
    def __init__(self, node_type: NodeType, object: Expression, body: Statement):
        super().__init__(node_type)
        self.object = object
        self.body = body
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.WITH_STMT:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.object):
            throw_invalid_type(self.object.node_type, self, attr="object")
        if not is_statement(self.body):
            throw_invalid_type(self.body.node_type, self, attr="body")

    def get_children(self) -> List[Node]:
        return [self.object, self.body]

    def get_children_names(self) -> List[str]:
        return ["object", "body"]
