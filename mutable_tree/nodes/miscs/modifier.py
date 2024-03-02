from ..node import Node, NodeType, NodeList
from ..utils import throw_invalid_type
from typing import List


class Modifier(Node):
    def __init__(self, node_type: NodeType, modifier: str):
        super().__init__(node_type)
        self.modifier = modifier
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.MODIFIER:
            throw_invalid_type(self.node_type, NodeType.MODIFIER)

    def get_children(self) -> List[Node]:
        return []

    def get_children_names(self) -> List[str]:
        return []


class ModifierList(NodeList):
    node_list: List[Modifier]

    def __init__(self, node_type: NodeType, modifiers: List[Modifier]):
        super().__init__(node_type)
        self.node_list = modifiers
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.MODIFIER_LIST:
            throw_invalid_type(self.node_type, NodeType.MODIFIER_LIST)
        for i, modifier in enumerate(self.node_list):
            if modifier.node_type != NodeType.MODIFIER:
                throw_invalid_type(
                    modifier.node_type, NodeType.MODIFIER, attr=f"modifier#{i}"
                )
