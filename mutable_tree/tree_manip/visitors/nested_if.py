import copy
from .visitor import TransformingVisitor
from mutable_tree.nodes import Node, node_factory
from mutable_tree.nodes import IfStatement
from mutable_tree.nodes import BinaryExpression, BinaryOps, ParenthesizedExpression
from typing import Optional


class NestedIfVisitor(TransformingVisitor):
    def visit_IfStatement(
        self,
        node: IfStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)

        cond = node.condition
        if not isinstance(cond, BinaryExpression):
            return (False, [])
        if cond.op != BinaryOps.AND:
            return (False, [])

        # if there is an alternate, we don't mess with it for simplicity
        if node.alternate is not None:
            return (False, [])

        cond_1 = cond.left
        cond_2 = cond.right

        # remove parentheses for better readability
        if isinstance(cond_1, ParenthesizedExpression):
            cond_1 = cond_1.expr
        if isinstance(cond_2, ParenthesizedExpression):
            cond_2 = cond_2.expr

        inner_if = node_factory.create_if_stmt(cond_2, node.consequence)
        inner_if_b = node_factory.wrap_block_stmt(inner_if)
        new_node = node_factory.create_if_stmt(cond_1, inner_if_b)

        return (True, [new_node])
