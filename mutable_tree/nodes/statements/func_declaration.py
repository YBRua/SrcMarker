from mutable_tree.nodes.node import NodeType
from ..node import Node, NodeType, NodeList
from ..miscs import ModifierList
from ..types import Dimensions, TypeIdentifierList, TypeParameterList
from ..utils import throw_invalid_type
from .statement import Statement
from .local_var_decl import DeclaratorType
from .declarators import Declarator, is_declarator
from .block_stmt import BlockStatement
from .empty_stmt import EmptyStatement
from typing import List, Optional, Union


class FormalParameter(Node):
    pass


def is_formal_parameter(node: Node) -> bool:
    return isinstance(node, FormalParameter)


class VariadicParameter(FormalParameter):
    def __init__(self, node_type: NodeType):
        super().__init__(node_type)
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.VARIADIC_PARAMETER:
            throw_invalid_type(self.node_type, self)


class UntypedParameter(FormalParameter):
    def __init__(self, node_type: NodeType, decl: Declarator):
        super().__init__(node_type)
        self.declarator = decl
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.UNTYPED_PARAMETER:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr="declarator")

    def get_children(self) -> List[Node]:
        return [self.declarator]

    def get_children_names(self) -> List[str]:
        return ["declarator"]


class TypedFormalParameter(FormalParameter):
    def __init__(
        self,
        node_type: NodeType,
        decl_type: DeclaratorType,
        decl: Optional[Declarator] = None,
    ):
        super().__init__(node_type)
        self.declarator = decl
        self.decl_type = decl_type
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FORMAL_PARAMETER:
            throw_invalid_type(self.node_type, self)
        if self.decl_type.node_type != NodeType.DECLARATOR_TYPE:
            throw_invalid_type(self.decl_type.node_type, self, attr="decl_type")
        if self.declarator is not None and not is_declarator(self.declarator):
            throw_invalid_type(self.declarator, self, attr="declarator")

    def get_children(self) -> List[Node]:
        if self.declarator is not None:
            return [self.decl_type, self.declarator]
        else:
            return [self.decl_type]

    def get_children_names(self) -> List[str]:
        return ["decl_type", "declarator"]


class SpreadParameter(FormalParameter):
    def __init__(
        self, node_type: NodeType, decl: Declarator, decl_type: DeclaratorType
    ):
        super().__init__(node_type)
        self.declarator = decl
        self.decl_type = decl_type
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.SPREAD_PARAMETER:
            throw_invalid_type(self.node_type, self)
        if self.decl_type.node_type != NodeType.DECLARATOR_TYPE:
            throw_invalid_type(self.decl_type.node_type, self, attr="decl_type")
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator, self, attr="declarator")

    def get_children(self) -> List[Node]:
        return [self.decl_type, self.declarator]

    def get_children_names(self) -> List[str]:
        return ["decl_type", "declarator"]


class FormalParameterList(NodeList):
    node_list: List[FormalParameter]

    def __init__(self, node_type: NodeType, parameters: List[FormalParameter]):
        super().__init__(node_type)
        self.node_list = parameters
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FORMAL_PARAMETER_LIST:
            throw_invalid_type(self.node_type, self)
        for i, param in enumerate(self.node_list):
            if not is_formal_parameter(param):
                throw_invalid_type(param.node_type, self, attr=f"param#{i}")


class FunctionDeclarator(Declarator):
    def __init__(
        self,
        node_type: NodeType,
        declarator: Declarator,
        parameters: FormalParameterList,
    ):
        super().__init__(node_type)
        self.declarator = declarator
        self.parameters = parameters
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FUNCTION_DECLARATOR:
            throw_invalid_type(self.node_type, self)
        if not is_declarator(self.declarator):
            throw_invalid_type(self.declarator.node_type, self, attr="name")
        if self.parameters.node_type != NodeType.FORMAL_PARAMETER_LIST:
            throw_invalid_type(self.parameters.node_type, self, attr="parameters")

    def get_children(self) -> List[Node]:
        return [self.declarator, self.parameters]

    def get_children_names(self) -> List[str]:
        return ["declarator", "parameters"]


class FunctionHeader(Node):
    def __init__(
        self,
        node_type: NodeType,
        func_decl: FunctionDeclarator,
        return_type: Optional[DeclaratorType] = None,
        dimensions: Optional[Dimensions] = None,
        throws: Optional[TypeIdentifierList] = None,
        modifiers: Optional[ModifierList] = None,
        type_params: Optional[TypeParameterList] = None,
    ):
        super().__init__(node_type)
        self.return_type = return_type
        self.func_decl = func_decl
        self.dimensions = dimensions
        self.throws = throws
        self.modifiers = modifiers
        self.type_params = type_params
        # TODO: dimensions for functions?
        if dimensions is not None:
            raise NotImplementedError("dimensions for functions are not supported yet")
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FUNCTION_HEADER:
            throw_invalid_type(self.node_type, self)
        if (
            self.return_type is not None
            and self.return_type.node_type != NodeType.DECLARATOR_TYPE
        ):
            throw_invalid_type(self.return_type.node_type, self, attr="return_type")
        if not is_declarator(self.func_decl):
            throw_invalid_type(self.func_decl.node_type, self, attr="name")
        if (
            self.dimensions is not None
            and self.dimensions.node_type != NodeType.DIMENSIONS
        ):
            throw_invalid_type(self.dimensions.node_type, self, attr="dimensions")
        if (
            self.throws is not None
            and self.throws.node_type != NodeType.TYPE_IDENTIFIER_LIST
        ):
            throw_invalid_type(self.throws.node_type, self, attr="throws")
        if (
            self.modifiers is not None
            and self.modifiers.node_type != NodeType.MODIFIER_LIST
        ):
            throw_invalid_type(self.modifiers.node_type, self, attr="modifiers")
        if (
            self.type_params is not None
            and self.type_params.node_type != NodeType.TYPE_PARAMETER_LIST
        ):
            throw_invalid_type(self.type_params.node_type, self, attr="type_params")

    def get_children(self) -> List[Node]:
        children = []
        if self.modifiers is not None:
            children.append(self.modifiers)
        if self.type_params is not None:
            children.append(self.type_params)
        children.append(self.return_type)
        children.append(self.func_decl)
        if self.dimensions is not None:
            children.append(self.dimensions)
        if self.throws is not None:
            children.append(self.throws)

        return children

    def get_children_names(self) -> List[str]:
        return [
            "modifiers",
            "type_params",
            "return_type",
            "func_decl",
            "dimensions",
            "throws",
        ]


class FunctionDeclaration(Statement):
    def __init__(
        self,
        node_type: NodeType,
        header: FunctionHeader,
        body: Union[BlockStatement, EmptyStatement],
    ):
        super().__init__(node_type)
        self.header = header
        self.body = body
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FUNCTION_DEFINITION:
            throw_invalid_type(self.node_type, self)
        if self.header.node_type != NodeType.FUNCTION_HEADER:
            throw_invalid_type(self.header.node_type, self, attr="header")
        if (
            self.body.node_type != NodeType.BLOCK_STMT
            and self.body.node_type != NodeType.EMPTY_STMT
        ):
            throw_invalid_type(self.body.node_type, self, attr="body")

    def get_children(self) -> List[Node]:
        return [self.header, self.body]

    def get_children_names(self) -> List[str]:
        return ["header", "body"]
