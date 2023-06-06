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
            throw_invalid_type(self.declarator.node_type, self, attr='declarator')
        if (not is_expression(self.value)
                and self.value.node_type != NodeType.EXPRESSION_LIST):
            throw_invalid_type(self.value.node_type, self, attr='value')

    def to_string(self) -> str:
        if is_expression(self.value):
            return f'{self.declarator.to_string()} = {self.value.to_string()}'
        else:
            arg_list = ', '.join(arg.to_string() for arg in self.value.get_children())
            return f'{self.declarator.to_string()}({arg_list})'

    def get_children(self) -> List[Node]:
        return [self.declarator, self.value]

    def get_children_names(self) -> List[str]:
        return ['declarator', 'value']


class VariableDeclarator(Declarator):

    def __init__(self, node_type: NodeType, decl_id: Identifier):
        super().__init__(node_type)
        self.decl_id = decl_id
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.VARIABLE_DECLARATOR:
            throw_invalid_type(self.node_type, self)

    def to_string(self) -> str:
        return self.decl_id.to_string()

    def get_children(self) -> List[Node]:
        return [self.decl_id]

    def get_children_names(self) -> List[str]:
        return ['decl_id']


class PointerDeclarator(Declarator):

    def __init__(self, node_type: NodeType, declarator: Declarator):
        super().__init__(node_type)
        self.declarator = declarator
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.POINTER_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr='declarator')

    def to_string(self) -> str:
        return f'*{self.declarator.to_string()}'

    def get_children(self) -> List[Node]:
        return [self.declarator]

    def get_children_names(self) -> List[str]:
        return ['declarator']


class ReferenceDeclarator(Declarator):

    def __init__(self, node_type: NodeType, declarator: Declarator, r_ref: bool = False):
        super().__init__(node_type)
        self.declarator = declarator
        self.r_ref = r_ref
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.REFERENCE_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr='declarator')

    def to_string(self) -> str:
        return f'{"&" if not self.r_ref else "&&"}{self.declarator.to_string()}'

    def get_children(self) -> List[Node]:
        return [self.declarator]

    def get_children_names(self) -> List[str]:
        return ['declarator']


class ArrayDeclarator(Declarator):

    def __init__(self, node_type: NodeType, declarator: Declarator,
                 dim: DimensionSpecifier):
        super().__init__(node_type)
        self.declarator = declarator
        self.dim = dim
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.ARRAY_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr='declarator')
        if self.dim.node_type != NodeType.DIMENSION_SPECIFIER:
            throw_invalid_type(self.dim.node_type, self, attr='dim')

    def to_string(self) -> str:
        return f'{self.declarator.to_string()}{self.dim.to_string()}'

    def get_children(self) -> List[Node]:
        return [self.declarator, self.dim]

    def get_children_names(self) -> List[str]:
        return ['declarator', 'dim']
