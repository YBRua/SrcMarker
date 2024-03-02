from ..node import Node, NodeType, NodeList
from .statement import Statement
from .block_stmt import BlockStatement
from ..expressions import Identifier, ExpressionList
from ..types import TypeIdentifierList
from ..miscs import ModifierList
from ..utils import throw_invalid_type
from typing import List, Optional, Union


class CatchClause(Node):
    def __init__(
        self,
        node_type: NodeType,
        body: BlockStatement,
        catch_types: Optional[TypeIdentifierList] = None,
        exception: Optional[Identifier] = None,
        modifiers: Optional[ModifierList] = None,
    ):
        super().__init__(node_type)
        self.catch_types = catch_types
        self.exception = exception
        self.body = body
        self.modifiers = modifiers
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.CATCH_CLAUSE:
            throw_invalid_type(self.node_type, self)
        if self.body.node_type != NodeType.BLOCK_STMT:
            throw_invalid_type(self.body.node_type, self, attr="body")
        if (
            self.catch_types is not None
            and self.catch_types.node_type != NodeType.TYPE_IDENTIFIER_LIST
        ):
            throw_invalid_type(self.catch_types.node_type, self, attr="catch_types")
        if (
            self.exception is not None
            and self.exception.node_type != NodeType.IDENTIFIER
        ):
            throw_invalid_type(self.exception.node_type, self, attr="exception")
        if (
            self.modifiers is not None
            and self.modifiers.node_type != NodeType.MODIFIER_LIST
        ):
            throw_invalid_type(self.modifiers.node_type, self, attr="modifiers")

    def get_children(self) -> List[Node]:
        if self.modifiers is not None:
            return [self.modifiers, self.catch_types, self.exception, self.body]
        else:
            return [self.catch_types, self.exception, self.body]

    def get_children_names(self) -> List[str]:
        return ["modifiers", "catch_types", "exception", "body"]


class TryHandlers(NodeList):
    node_list: List[CatchClause]

    def __init__(self, node_type: NodeType, handlers: List[CatchClause]):
        super().__init__(node_type)
        self.node_list = handlers

    def _check_types(self):
        if self.node_type != NodeType.TRY_HANDLERS:
            throw_invalid_type(self.node_type, self)
        for i, handler in enumerate(self.node_list):
            if handler.node_type != NodeType.CATCH_CLAUSE:
                throw_invalid_type(handler.node_type, self, attr=f"handler#{i}")


class FinallyClause(Node):
    def __init__(self, node_type: NodeType, body: BlockStatement):
        super().__init__(node_type)
        self.body = body
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.FINALLY_CLAUSE:
            throw_invalid_type(self.node_type, self)
        if self.body.node_type != NodeType.BLOCK_STMT:
            throw_invalid_type(self.body.node_type, self, attr="body")

    def get_children_names(self) -> List[str]:
        return ["body"]


class TryStatement(Statement):
    def __init__(
        self,
        node_type: NodeType,
        body: BlockStatement,
        handlers: Optional[TryHandlers] = None,
        finalizer: Optional[FinallyClause] = None,
    ):
        super().__init__(node_type)
        self.body = body
        self.handlers = handlers
        self.finalizer = finalizer
        self._check_types()

    def _check_types(self):
        if self.node_type != NodeType.TRY_STMT:
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

    def get_children_names(self) -> List[str]:
        return ["body", "handlers", "finalizer"]
