from ..node import Node, NodeType, NodeList
from .statement import Statement
from .block_stmt import BlockStatement
from .try_stmt import TryHandlers, FinallyClause
from .local_var_decl import LocalVariableDeclaration
from ..expressions import is_expression
from ..expressions import Identifier, FieldAccess
from ..utils import throw_invalid_type
from typing import List, Optional, Union


class TryResource(Node):
    def __init__(
        self,
        node_type: NodeType,
        resource: Union[LocalVariableDeclaration, Identifier, FieldAccess],
    ):
        super().__init__(node_type)
        self.resource = resource
        self._check_types

    def _check_types(self):
        if self.node_type != NodeType.TRY_RESOURCE:
            throw_invalid_type(self.node_type, self)

        res_type = self.resource.node_type
        if res_type not in {
            NodeType.IDENTIFIER,
            NodeType.FIELD_ACCESS,
            NodeType.LOCAL_VARIABLE_DECLARATION,
        }:
            throw_invalid_type(res_type, self, attr="resource")

    def get_children_names(self) -> List[str]:
        return ["resource"]

    def get_children(self) -> List[Node]:
        return [self.resource]


class TryResourceList(NodeList):
    node_list: List[TryResource]

    def __init__(self, node_type: NodeType, resources: List[TryResource]):
        super().__init__(node_type)
        self.node_list = resources

    def _check_types(self):
        if self.node_type != NodeType.TRY_RESOURCE_LIST:
            throw_invalid_type(self.node_type, self)
        for i, resource in enumerate(self.node_list):
            if resource.node_type != NodeType.TRY_RESOURCE:
                throw_invalid_type(resource.node_type, self, attr=f"resource#{i}")


class TryWithResourcesStatement(Statement):
    def __init__(
        self,
        node_type: NodeType,
        resources: TryResourceList,
        body: BlockStatement,
        handlers: Optional[TryHandlers] = None,
        finalizer: Optional[FinallyClause] = None,
    ):
        super().__init__(node_type)
        self.body = body
        self.resources = resources
        self.handlers = handlers
        self.finalizer = finalizer
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.TRY_WITH_RESOURCES_STMT:
            throw_invalid_type(self.node_type, self)
        if self.body.node_type != NodeType.BLOCK_STMT:
            throw_invalid_type(self.body.node_type, self, attr="body")
        if (
            self.finalizer is not None
            and self.finalizer.node_type != NodeType.FINALLY_CLAUSE
        ):
            throw_invalid_type(self.finalizer.node_type, self, attr="finalizer")
        if (
            self.handlers is not None
            and self.handlers.node_type != NodeType.TRY_HANDLERS
        ):
            throw_invalid_type(self.handlers.node_type, self, attr="handlers")
        if self.resources.node_type != NodeType.TRY_RESOURCE_LIST:
            throw_invalid_type(self.resources.node_type, self, attr="resources")

    def get_children_names(self) -> List[str]:
        return ["resources", "body", "handlers", "finalizer"]

    def get_children(self) -> List[Node]:
        children = [self.resources, self.body]
        if self.handlers is not None:
            children.append(self.handlers)
        if self.finalizer is not None:
            children.append(self.finalizer)
        return children
