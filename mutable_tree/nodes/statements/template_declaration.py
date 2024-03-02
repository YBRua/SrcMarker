from enum import Enum
from ..node import Node, NodeType, NodeList
from ..types import TypeIdentifier
from ..utils import throw_invalid_type
from .statement import Statement
from .func_declaration import FormalParameter, is_formal_parameter, FunctionDeclaration
from typing import List, Union


class TypenameOpts(Enum):
    CLASS = "class"
    TYPENAME = "typename"


def get_typename_opts(tn: str) -> TypenameOpts:
    if tn == "class":
        return TypenameOpts.CLASS
    elif tn == "typename":
        return TypenameOpts.TYPENAME
    else:
        raise ValueError(f"Invalid typename: {tn}")


class TypeParameterDeclaration(Node):
    def __init__(
        self, node_type: NodeType, type_id: TypeIdentifier, typename_opt: TypenameOpts
    ):
        super().__init__(node_type)
        self.type_id = type_id
        self.typename_opt = typename_opt
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.TYPE_PARAMETER_DECLARATION:
            throw_invalid_type(self.node_type, self)
        if self.type_id.node_type != NodeType.TYPE_IDENTIFIER:
            throw_invalid_type(self.type_id.node_type, self, attr="type_id")


TemplateParameter = Union[TypeParameterDeclaration, FormalParameter]


class TemplateParameterList(NodeList):
    node_list: List[TemplateParameter]

    def __init__(self, node_type: NodeType, params: List[TemplateParameter]):
        super().__init__(node_type)
        self.node_list = params
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.TEMPLATE_PARAMETER_LIST:
            throw_invalid_type(self.node_type, self)
        for i, param in enumerate(self.node_list):
            if (
                not is_formal_parameter(param)
                and param.node_type != NodeType.TYPE_PARAMETER_DECLARATION
            ):
                throw_invalid_type(param.node_type, self, attr=f"param#{i}")


class TemplateDeclaration(Statement):
    def __init__(
        self,
        node_type: NodeType,
        params: TemplateParameterList,
        func_decl: FunctionDeclaration,
    ):
        super().__init__(node_type)
        self.params = params
        self.func_decl = func_decl
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.TEMPLATE_DECLARATION:
            throw_invalid_type(self.node_type, self)
        if self.params.node_type != NodeType.TEMPLATE_PARAMETER_LIST:
            throw_invalid_type(self.params.node_type, self, attr="params")
        if self.func_decl.node_type != NodeType.FUNCTION_DEFINITION:
            throw_invalid_type(self.func_decl.node_type, self, attr="func_decl")

    def get_children(self) -> List[Node]:
        return [self.params, self.func_decl]

    def get_children_names(self) -> List[str]:
        return ["params", "func_decl"]
