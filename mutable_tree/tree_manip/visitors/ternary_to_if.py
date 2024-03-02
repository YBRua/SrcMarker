import copy
from .visitor import TransformingVisitor
from mutable_tree.nodes import Node
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import AssignmentOps
from mutable_tree.nodes import (
    Expression,
    ExpressionStatement,
    AssignmentExpression,
    TernaryExpression,
)
from typing import Optional


class TernaryToIfVisitor(TransformingVisitor):
    def _wrap_assignment_block(self, lhs: Expression, rhs: Expression):
        lhs = copy.deepcopy(lhs)
        expr = node_factory.create_assignment_expr(lhs, rhs, AssignmentOps.EQUAL)
        stmt = node_factory.create_expression_stmt(expr)
        return node_factory.create_block_stmt(
            node_factory.create_statement_list([stmt])
        )

    def visit_ExpressionStatement(
        self,
        node: ExpressionStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        expr = node.expr
        if not isinstance(expr, AssignmentExpression):
            return (False, [])
        if expr.op != AssignmentOps.EQUAL:
            return (False, [])

        lhs = expr.left
        rhs = expr.right
        if not isinstance(rhs, TernaryExpression):
            return (False, [])

        condition = rhs.condition
        consequence = rhs.consequence
        alternative = rhs.alternative

        consequence_block = self._wrap_assignment_block(lhs, consequence)
        alternative_block = self._wrap_assignment_block(lhs, alternative)
        new_node = node_factory.create_if_stmt(
            condition, consequence_block, alternative_block
        )

        # also transform the new if statement
        self.generic_visit(new_node, parent, parent_attr)

        return (True, [new_node])
