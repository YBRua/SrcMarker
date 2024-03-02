from ..node import Node, NodeType
from .dimensions import Dimensions
from typing import List, Optional


class TypeIdentifier(Node):
    def __init__(
        self,
        node_type: NodeType,
        type_identifier: str,
        dimension: Optional[Dimensions] = None,
    ):
        super().__init__(node_type)
        self.type_identifier = type_identifier
        self.dimension = dimension

    def _check_types(self):
        if self.node_type != NodeType.TYPE_IDENTIFIER:
            raise TypeError(f"Invalid type: {self.node_type} for TypeIdentifier.")

    def get_children(self) -> List[Node]:
        if self.dimension is not None:
            return [self.dimension]
        else:
            return []

    def get_children_names(self) -> List[str]:
        return ["dimension"]
