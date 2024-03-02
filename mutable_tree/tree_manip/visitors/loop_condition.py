from .visitor import TransformingVisitor
from mutable_tree.nodes import Node
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import WhileStatement, ForStatement, Literal
from typing import Optional


class LoopLiteralOneVisitor(TransformingVisitor):
    def visit_WhileStatement(
        self,
        node: WhileStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)
        condition = node.condition
        if isinstance(condition, Literal) and condition.value == "true":
            node.condition = node_factory.create_literal("1")
            return (True, [node])

        return (False, [])

    def visit_ForStatement(
        self,
        node: ForStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)
        condition = node.condition
        if (
            isinstance(condition, Literal) and condition.value == "true"
        ) or condition is None:
            node.condition = node_factory.create_literal("1")
            return (True, [node])

        return (False, [])


class LoopLiteralTrueVisitor(TransformingVisitor):
    def visit_WhileStatement(
        self,
        node: WhileStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)
        condition = node.condition
        if isinstance(condition, Literal) and condition.value == "1":
            node.condition = node_factory.create_literal("true")
            return (True, [node])

        return (False, [])

    def visit_ForStatement(
        self,
        node: ForStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)
        condition = node.condition
        if (
            isinstance(condition, Literal) and condition.value == "1"
        ) or condition is None:
            node.condition = node_factory.create_literal("true")
            return (True, [node])

        return (False, [])
