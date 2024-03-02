from ..node import Node, NodeType
from ..expressions import Expression, ExpressionList, Identifier
from ..types import DimensionSpecifier
from ..expressions import is_expression
from ..utils import throw_invalid_type
from typing import List, Union


class Declarator(Node):
    pass


def is_declarator(node: Node) -> bool:
    return isinstance(node, Declarator)


class InitializingDeclarator(Declarator):
    def __init__(
        self,
        node_type: NodeType,
        declarator: Declarator,
        value: Union[Expression, ExpressionList],
    ):
        super().__init__(node_type)
        self.declarator = declarator
        self.value = value
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.INITIALIZING_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr="declarator")
        if (
            not is_expression(self.value)
            and self.value.node_type != NodeType.EXPRESSION_LIST
            and self.value.node_type != NodeType.FUNCTION_DEFINITION
        ):
            # FIXME: function definition is not an expression
            throw_invalid_type(self.value.node_type, self, attr="value")

    def get_children(self) -> List[Node]:
        return [self.declarator, self.value]

    def get_children_names(self) -> List[str]:
        return ["declarator", "value"]


class DestructuringDeclarator(Declarator):
    def __init__(self, node_type: NodeType, pattern: Expression):
        super().__init__(node_type)
        self.pattern = pattern
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.DESTRUCTURING_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_expression(self.pattern):
            throw_invalid_type(self.pattern.node_type, self, attr="pattern")

    def get_children(self) -> List[Node]:
        return [self.pattern]

    def get_children_names(self) -> List[str]:
        return ["pattern"]


class VariableDeclarator(Declarator):
    def __init__(self, node_type: NodeType, decl_id: Identifier):
        super().__init__(node_type)
        self.decl_id = decl_id
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.VARIABLE_DECLARATOR:
            throw_invalid_type(self.node_type, self)

    def get_children(self) -> List[Node]:
        return [self.decl_id]

    def get_children_names(self) -> List[str]:
        return ["decl_id"]


class AnonymousDeclarator(Declarator):
    def __init__(self, node_type: NodeType):
        super().__init__(node_type)
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.ANONYMOUS_DECLARATOR:
            throw_invalid_type(self.node_type, self)

    def get_children(self) -> List[Node]:
        return []

    def get_children_names(self) -> List[str]:
        return []


class PointerDeclarator(Declarator):
    def __init__(self, node_type: NodeType, declarator: Declarator):
        super().__init__(node_type)
        self.declarator = declarator
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.POINTER_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr="declarator")

    def get_children(self) -> List[Node]:
        return [self.declarator]

    def get_children_names(self) -> List[str]:
        return ["declarator"]


class ReferenceDeclarator(Declarator):
    def __init__(
        self, node_type: NodeType, declarator: Declarator, r_ref: bool = False
    ):
        super().__init__(node_type)
        self.declarator = declarator
        self.r_ref = r_ref
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.REFERENCE_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr="declarator")

    def get_children(self) -> List[Node]:
        return [self.declarator]

    def get_children_names(self) -> List[str]:
        return ["declarator"]


class ArrayDeclarator(Declarator):
    def __init__(
        self, node_type: NodeType, declarator: Declarator, dim: DimensionSpecifier
    ):
        super().__init__(node_type)
        self.declarator = declarator
        self.dim = dim
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.ARRAY_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr="declarator")
        if self.dim.node_type != NodeType.DIMENSION_SPECIFIER:
            throw_invalid_type(self.dim.node_type, self, attr="dim")

    def get_children(self) -> List[Node]:
        return [self.declarator, self.dim]

    def get_children_names(self) -> List[str]:
        return ["declarator", "dim"]
