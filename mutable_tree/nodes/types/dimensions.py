from ..node import Node, NodeType, NodeList
from ..expressions import Expression, is_expression
from ..utils import throw_invalid_type
from typing import List, Optional


class DimensionSpecifier(Node):
    def __init__(self, node_type: NodeType, expr: Optional[Expression] = None):
        super().__init__(node_type)
        self.expr = expr
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.DIMENSION_SPECIFIER:
            throw_invalid_type(self.node_type, self)
        if self.expr is not None and not is_expression(self.expr):
            throw_invalid_type(self.expr.node_type, self, "expr")

    def get_children(self) -> List[Node]:
        if self.expr is not None:
            return [self.expr]
        else:
            return []

    def get_children_names(self) -> List[str]:
        return ["expr"]


class Dimensions(NodeList):
    node_list: List[DimensionSpecifier]

    def __init__(self, node_type: NodeType, dims: List[DimensionSpecifier]):
        super().__init__(node_type)
        self.node_list = dims
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.DIMENSIONS:
            throw_invalid_type(self.node_type, self)
        for i, dim in enumerate(self.node_list):
            if dim.node_type != NodeType.DIMENSION_SPECIFIER:
                throw_invalid_type(dim.node_type, self, f"dim#{i}")
