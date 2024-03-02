from ..node import Node, NodeType
from ..utils import throw_invalid_type
from .type_identifier_list import TypeIdentifierList
from typing import List, Optional


class TypeParameter(Node):
    def __init__(
        self,
        node_type: NodeType,
        type_identifier: str,
        extends: Optional[TypeIdentifierList] = None,
    ):
        super().__init__(node_type)
        self.type_identifier = type_identifier
        self.extends = extends

    def _check_types(self):
        if self.node_type != NodeType.TYPE_PARAMETER:
            throw_invalid_type(self.node_type, self)
        if (
            self.extends is not None
            and self.extends.node_type != NodeType.TYPE_IDENTIFIER_LIST
        ):
            throw_invalid_type(self.extends.node_type, self, attr="extends")

    def get_children(self) -> List[Node]:
        if self.extends is not None:
            return [self.extends]
        else:
            return []

    def get_children_names(self) -> List[str]:
        return ["extends"]
