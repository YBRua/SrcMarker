from .visitor import TransformingVisitor
from mutable_tree.nodes import Node, NodeType, node_factory
from mutable_tree.nodes import IfStatement, BlockStatement
from mutable_tree.nodes import Expression, BinaryExpression, BinaryOps
from typing import Optional


class CompoundIfVisitor(TransformingVisitor):
    def _create_logical_and(self, lhs: Expression, rhs: Expression) -> BinaryExpression:
        singleton_types = {
            NodeType.LITERAL,
            NodeType.IDENTIFIER,
            NodeType.CALL_EXPR,
            NodeType.PARENTHESIZED_EXPR,
        }
        if lhs.node_type not in singleton_types:
            lhs = node_factory.create_parenthesized_expr(lhs)
        if rhs.node_type not in singleton_types:
            rhs = node_factory.create_parenthesized_expr(rhs)
        return node_factory.create_binary_expr(lhs, rhs, BinaryOps.AND)

    def _find_nested_if(self, node: IfStatement) -> Optional[IfStatement]:
        # only transform if stmts s.t.
        # 1. has no else
        if node.alternate is not None:
            return None

        # 2. has a single if in its body
        body = node.consequence
        if isinstance(body, BlockStatement):
            stmts = body.stmts.get_children()
            # single child
            if len(stmts) != 1:
                return None
            # single child is if
            if not isinstance(stmts[0], IfStatement):
                return None
            # single if-node has no else
            if stmts[0].alternate is not None:
                return None
            nested_if = stmts[0]
        elif not isinstance(body, IfStatement):
            return None
        else:
            nested_if = body

        # return the if statement if it is a valid candidate
        return nested_if

    def visit_IfStatement(
        self,
        node: IfStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)

        candidate = self._find_nested_if(node)
        if candidate is None:
            return (False, [])

        # apply transform
        cond_1 = node.condition
        cond_2 = candidate.condition

        if_cond = self._create_logical_and(cond_1, cond_2)
        if_body = candidate.consequence
        if_alt = candidate.alternate
        if_stmt = node_factory.create_if_stmt(if_cond, if_body, if_alt)

        return (True, [if_stmt])
