from .expression import Expression
from .identifier import Identifier
from ..types import TypeIdentifier
from ..utils import throw_invalid_type
from ..node import Node, NodeType
from typing import List, Optional, Union


class ScopeResolution(Node):
    def __init__(
        self,
        node_type: NodeType,
        scope: Optional[Union[Identifier, TypeIdentifier]] = None,
    ):
        super().__init__(node_type)
        self.scope = scope
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.SCOPE_RESOLUTION:
            throw_invalid_type(self.node_type, self)
        if (
            self.scope is not None
            and self.scope.node_type != NodeType.IDENTIFIER
            and self.scope.node_type != NodeType.TYPE_IDENTIFIER
        ):
            throw_invalid_type(self.scope.node_type, self, attr="scope")

    def get_children(self) -> List[Node]:
        if self.scope is None:
            return []
        return [self.scope]

    def get_children_names(self) -> List[str]:
        return ["scope"]


class QualifiedIdentifier(Expression):
    def __init__(
        self,
        node_type: NodeType,
        scope: ScopeResolution,
        name: Union[Identifier, "QualifiedIdentifier", TypeIdentifier],
    ):
        super().__init__(node_type)
        self.scope = scope
        self.name = name
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.QUALIFIED_IDENTIFIER:
            throw_invalid_type(self.node_type, self)
        if self.scope.node_type != NodeType.SCOPE_RESOLUTION:
            throw_invalid_type(self.scope.node_type, self, attr="scope")
        if (
            self.name.node_type != NodeType.IDENTIFIER
            and self.name.node_type != NodeType.QUALIFIED_IDENTIFIER
            and self.name.node_type != NodeType.TYPE_IDENTIFIER
        ):
            throw_invalid_type(self.name.node_type, self, attr="name")

    def get_children(self) -> List[Node]:
        return [self.scope, self.name]

    def get_children_names(self) -> List[str]:
        return ["scope", "name"]
