from ..node import NodeType, NodeList
from ..utils import throw_invalid_type
from .type_parameter import TypeParameter
from typing import List


class TypeParameterList(NodeList):
    node_list: List[TypeParameter]

    def __init__(self, node_type: NodeType, type_params: List[TypeParameter]):
        super().__init__(node_type)
        self.node_list = type_params
        self._check_types

    def _check_types(self):
        if self.node_type != NodeType.TYPE_PARAMETER_LIST:
            throw_invalid_type(self.node_type, self)
        for i, type_id in enumerate(self.node_list):
            if type_id.node_type != NodeType.TYPE_PARAMETER:
                throw_invalid_type(type_id.node_type, self, f"type_param#{i}")
