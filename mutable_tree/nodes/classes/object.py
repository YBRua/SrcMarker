from ..node import Node, NodeType, NodeList
from ..expressions import Expression, is_expression
from ..expressions import Identifier, Literal, SpreadElement
from ..statements import FunctionDeclaration
from ..utils import throw_invalid_type
from typing import List, Union


class ComputedPropertyName(Node):
    def __init__(self, node_type: NodeType, expr: Expression):
        super().__init__(node_type)
        self.expr = expr

    def _check_types(self):
        if self.node_type != NodeType.COMPUTED_PROPERTY_NAME:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.expr):
            throw_invalid_type(self.expr.node_type, self, "expr")

    def get_children(self) -> List[Node]:
        return [self.expr]

    def get_children_names(self) -> List[str]:
        return ["expr"]


class KeyValuePair(Node):
    def __init__(
        self,
        node_type: NodeType,
        key: Union[Identifier, Literal, ComputedPropertyName],
        value: Expression,
    ):
        # NOTE: value should be a pattern
        # but we ignore the differences for the moment
        super().__init__(node_type)
        self.key = key
        self.value = value
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.KEYVALUE_PAIR:
            throw_invalid_type(self.node_type, self)
        if self.key.node_type not in {
            NodeType.IDENTIFIER,
            NodeType.LITERAL,
            NodeType.COMPUTED_PROPERTY_NAME,
        }:
            throw_invalid_type(self.key.node_type, self, "key")
        if (
            not is_expression(self.value)
            and self.value.node_type != NodeType.FUNCTION_DEFINITION
        ):
            throw_invalid_type(self.value.node_type, self, "value")

    def get_children(self) -> List[Node]:
        return [self.key, self.value]

    def get_children_names(self) -> List[str]:
        return ["key", "value"]


ObjectMember = Union[KeyValuePair, SpreadElement, FunctionDeclaration]


class ObjectMembers(NodeList):
    def __init__(self, node_type: NodeType, members: List[ObjectMember]):
        super().__init__(node_type)
        self.node_list = members
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.OBJECT_MEMBERS:
            throw_invalid_type(self.node_type, self)

        for i, member in enumerate(self.node_list):
            if member.node_type not in {
                NodeType.KEYVALUE_PAIR,
                NodeType.SPREAD_ELEMENT,
                NodeType.FUNCTION_DEFINITION,
                NodeType.IDENTIFIER,
            }:
                throw_invalid_type(member.node_type, self, f"member#{i}")


class Object(Expression):
    def __init__(self, node_type: NodeType, members: ObjectMembers):
        super().__init__(node_type)
        self.members = members
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.OBJECT:
            throw_invalid_type(self.node_type, self)
        if self.members.node_type != NodeType.OBJECT_MEMBERS:
            throw_invalid_type(self.members.node_type, self, "members")

    def get_children(self) -> List[Node]:
        return [self.members]

    def get_children_names(self) -> List[str]:
        return ["members"]
