import copy
from .visitor import TransformingVisitor
from mutable_tree.nodes import Node, node_factory
from mutable_tree.nodes import BinaryOps
from mutable_tree.nodes import (
    SwitchStatement,
    SwitchCaseList,
    SwitchCase,
    BlockStatement,
    BreakStatement,
)
from typing import Optional


class SwitchToIfVisitor(TransformingVisitor):
    def _can_transform(self, cases: SwitchCaseList):
        # all cases must end with a break statement
        # except the default case (at the end)
        c: SwitchCase
        n_cases = len(cases.get_children())
        for i, c in enumerate(cases.get_children()):
            if i == n_cases - 1 and c.case is None:
                # last case is default case
                continue
            # check last statement
            stmts = c.stmts.get_children()
            if len(stmts) == 0:
                return False
            last_stmt = stmts[-1]
            if isinstance(last_stmt, BlockStatement):
                # assume at most one block in the case
                last_stmt = last_stmt.stmts.get_child_at(-1)
            if not isinstance(last_stmt, BreakStatement):
                return False
        return True

    def _remove_break_stmts(self, cases: SwitchCaseList):
        c: SwitchCase
        for c in cases.get_children():
            stmts = c.stmts.get_children()

            if len(stmts) == 0:
                return

            if isinstance(stmts[-1], BreakStatement):
                stmts.pop()

            elif isinstance(stmts[-1], BlockStatement):
                # assume at most one block in the case
                stmts = stmts[-1].stmts.get_children()
                if isinstance(stmts[-1], BreakStatement):
                    stmts.pop()

    def visit_SwitchStatement(
        self,
        node: SwitchStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)

        condition = node.condition
        cases = node.cases

        if not self._can_transform(cases):
            return (False, [])

        self._remove_break_stmts(cases)

        # apply transform
        c: SwitchCase
        initial = None
        prev = None
        final_else = None
        for c in cases.get_children():
            if c.case is None:
                # default case should be put at the end
                final_else = node_factory.create_block_stmt(c.stmts)
                continue

            # create if statement
            cond = copy.deepcopy(condition)
            if_cond = node_factory.create_binary_expr(cond, c.case, BinaryOps.EQ)
            if len(c.stmts.get_children()) == 1 and isinstance(
                c.stmts.get_child_at(0), BlockStatement
            ):
                if_body = c.stmts.get_child_at(0)
            else:
                if_body = node_factory.create_block_stmt(c.stmts)
            if_stmt = node_factory.create_if_stmt(if_cond, if_body)

            if prev is not None:
                prev.alternate = if_stmt
            else:
                initial = if_stmt

            prev = if_stmt

        # add final else
        if prev is not None:
            prev.alternate = final_else
        else:
            # some strange people only write a default case in switch
            initial = node_factory.create_if_stmt(condition, final_else)

        return (True, [initial])
