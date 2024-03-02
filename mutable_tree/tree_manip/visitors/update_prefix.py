from .visitor import TransformingVisitor
from mutable_tree.nodes import Node
from mutable_tree.nodes import Literal, BinaryExpression
from mutable_tree.nodes import AssignmentOps, BinaryOps, UpdateOps
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import UpdateExpression, AssignmentExpression
from mutable_tree.stringifiers import BaseStringifier
from typing import Optional


class PrefixUpdateVisitor(TransformingVisitor):
    assign_op_to_update_op = {
        AssignmentOps.PLUS_EQUAL: UpdateOps.INCREMENT,
        AssignmentOps.MINUS_EQUAL: UpdateOps.DECREMENT,
    }
    binop_to_update_op = {
        BinaryOps.PLUS: UpdateOps.INCREMENT,
        BinaryOps.MINUS: UpdateOps.DECREMENT,
    }

    def visit_UpdateExpression(
        self,
        expr: UpdateExpression,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(expr, parent, parent_attr)

        if not expr.prefix:
            new_node = node_factory.create_update_expr(expr.operand, expr.op, True)
            return (True, [new_node])
        else:
            return (False, None)

    def visit_AssignmentExpression(
        self,
        expr: AssignmentExpression,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(expr, parent, parent_attr)

        if expr.op == AssignmentOps.PLUS_EQUAL or expr.op == AssignmentOps.MINUS_EQUAL:
            # i += 1
            rhs = expr.right
            # FIXME: values for literals are currently always strings
            if isinstance(rhs, Literal) and rhs.value == "1":
                update_op = self.assign_op_to_update_op[expr.op]
                new_node = node_factory.create_update_expr(expr.left, update_op, True)
                return (True, [new_node])

        if expr.op == AssignmentOps.EQUAL and isinstance(expr.right, BinaryExpression):
            # i = i + 1
            # NOTE: this visitor do not convert i = 1 + i
            stringifier = BaseStringifier()
            lhs_str = stringifier.stringify(expr.left)

            bin_expr = expr.right
            binop = bin_expr.op
            bin_lhs = bin_expr.left
            bin_lhs_str = stringifier.stringify(bin_lhs)
            bin_rhs = bin_expr.right
            if binop == BinaryOps.PLUS or binop == BinaryOps.MINUS:
                if (lhs_str == bin_lhs_str) and (
                    isinstance(bin_rhs, Literal) and bin_rhs.value == "1"
                ):
                    update_op = self.binop_to_update_op[binop]
                    new_node = node_factory.create_update_expr(
                        expr.left, update_op, True
                    )
                    return (True, [new_node])

        return (False, None)
