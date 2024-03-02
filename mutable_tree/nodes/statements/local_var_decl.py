from ..node import Node, NodeType, NodeList
from .statement import Statement
from .declarators import Declarator, is_declarator
from ..miscs import ModifierList
from ..types import TypeIdentifier
from ..utils import throw_invalid_type
from typing import List, Optional


class DeclaratorType(Node):
    def __init__(
        self,
        node_type: NodeType,
        type_id: TypeIdentifier,
        prefix_modifiers: Optional[ModifierList] = None,
        postfix_modifiers: Optional[ModifierList] = None,
    ):
        super().__init__(node_type)
        self.type_id = type_id
        self.prefix_modifiers = prefix_modifiers
        self.postfix_modifiers = postfix_modifiers
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.DECLARATOR_TYPE:
            throw_invalid_type(self.node_type, self)
        if (
            self.type_id.node_type != NodeType.TYPE_IDENTIFIER
            and self.type_id.node_type != NodeType.QUALIFIED_IDENTIFIER
        ):
            throw_invalid_type(self.type_id.node_type, self, attr="type_id")
        if (
            self.prefix_modifiers is not None
            and self.prefix_modifiers.node_type != NodeType.MODIFIER_LIST
        ):
            throw_invalid_type(
                self.prefix_modifiers.node_type, self, attr="prefix_modifiers"
            )
        if (
            self.postfix_modifiers is not None
            and self.postfix_modifiers.node_type != NodeType.MODIFIER_LIST
        ):
            throw_invalid_type(
                self.postfix_modifiers.node_type, self, attr="postfix_modifiers"
            )

    def get_children(self) -> List[Node]:
        children = [self.prefix_modifiers, self.type_id, self.postfix_modifiers]
        children = [child for child in children if child is not None]
        return children

    def get_children_names(self) -> List[str]:
        return ["prefix_modifiers", "type_id", "postfix_modifiers"]


class DeclaratorList(NodeList):
    node_list: List[Declarator]

    def __init__(self, node_type: NodeType, declarators: List[Declarator]):
        super().__init__(node_type)
        self.node_list = declarators
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.DECLARATOR_LIST:
            throw_invalid_type(self.node_type, self)
        for i, decl in enumerate(self.node_list):
            if not is_declarator(decl):
                throw_invalid_type(decl.node_type, self, attr=f"declarator#{i}")


class LocalVariableDeclaration(Statement):
    def __init__(
        self,
        node_type: NodeType,
        decl_type: DeclaratorType,
        declarators: DeclaratorList,
    ):
        super().__init__(node_type)
        self.type = decl_type
        self.declarators = declarators
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.LOCAL_VARIABLE_DECLARATION:
            throw_invalid_type(self.node_type, self)
        if self.type.node_type != NodeType.DECLARATOR_TYPE:
            throw_invalid_type(self.type.node_type, self, attr="type")
        if self.declarators.node_type != NodeType.DECLARATOR_LIST:
            throw_invalid_type(self.declarators.node_type, self, attr="declarators")

    def get_children(self) -> List[Node]:
        return [self.type, self.declarators]

    def get_children_names(self) -> List[str]:
        return ["type", "declarators"]
