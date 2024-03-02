from ..node import NodeType, NodeList
from ..utils import throw_invalid_type
from .type_identifier import TypeIdentifier
from typing import List


class TypeIdentifierList(NodeList):
    node_list: List[TypeIdentifier]

    def __init__(self, node_type: NodeType, type_ids: List[TypeIdentifier]):
        super().__init__(node_type)
        self.node_list = type_ids
        self._check_types

    def _check_types(self):
        if self.node_type != NodeType.TYPE_IDENTIFIER_LIST:
            throw_invalid_type(self.node_type, self)
        for i, type_id in enumerate(self.node_list):
            if type_id.node_type != NodeType.TYPE_IDENTIFIER:
                throw_invalid_type(type_id.node_type, self, f"type_id#{i}")
