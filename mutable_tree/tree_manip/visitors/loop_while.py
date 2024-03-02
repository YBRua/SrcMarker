from .visitor import TransformingVisitor
from mutable_tree.nodes import Node, NodeType, NodeList
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import (
    ForStatement,
    WhileStatement,
    Statement,
    ContinueStatement,
)
from mutable_tree.nodes import is_expression
from typing import Optional, List


class ForToWhileVisitor(TransformingVisitor):
    def _wrap_loop_body(self, body_stmts: List[Statement]):
        # remove empty statements
        for i, stmt in enumerate(body_stmts):
            if stmt.node_type == NodeType.EMPTY_STMT:
                body_stmts.pop(i)

        # if statement list is empty, return empty statement
        if len(body_stmts) == 0:
            return node_factory.create_empty_stmt()
        else:
            stmt_list = node_factory.create_statement_list(body_stmts)
            return node_factory.create_block_stmt(stmt_list)

    def _collect_continue_stmts(self, node: Node, parent: Optional[Node] = None):
        if isinstance(node, ContinueStatement):
            return [(node, parent)]
        else:
            continue_stmts = []
            for child_attr in node.get_children_names():
                child = node.get_child_at(child_attr)
                # note that we only collect continue statements in the same scope
                if (
                    child is not None
                    and not isinstance(child, ForStatement)
                    and not isinstance(child, WhileStatement)
                ):
                    continue_stmts += self._collect_continue_stmts(child, node)
            return continue_stmts

    def visit_ForStatement(
        self,
        node: ForStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)
        new_stmts = []
        init = node.init
        condition = node.condition
        update = node.update
        body = node.body

        # move init stmts to before loop
        if init is not None:
            if init.node_type == NodeType.EXPRESSION_LIST:
                # if init are expressions, upgrade to standalone statements
                init_exprs = init.get_children()
                for init_expr in init_exprs:
                    new_stmts.append(node_factory.create_expression_stmt(init_expr))
            else:
                new_stmts.append(init)

        # extract condition, update and body
        if condition is None:
            condition = node_factory.create_literal("true")
        if update is not None:
            update_exprs = update.get_children()
        else:
            update_exprs = []
        if body.node_type != NodeType.BLOCK_STMT:
            body_stmts = [body]
        else:
            # get statement list from block statement
            body_stmts = body.get_children()[0].get_children()

        # convert update expressions to statements and append to loop body
        for u in update_exprs:
            assert is_expression(u)
            body_stmts.append(node_factory.create_expression_stmt(u))

        # collect all continue statements and add the updates before continues
        continues = self._collect_continue_stmts(body)
        for cont_node, cont_parent in continues:
            if isinstance(cont_parent, NodeList):
                new_node_list = []
                for child_attr in cont_parent.get_children_names():
                    child = cont_parent.get_child_at(child_attr)
                    if child is not None:
                        if child is cont_node:
                            for u in update_exprs:
                                assert is_expression(u)
                                new_node_list.append(
                                    node_factory.create_expression_stmt(u)
                                )
                            new_node_list.append(cont_node)
                        else:
                            new_node_list.append(child)
                cont_parent.node_list = new_node_list
            else:
                continue_block_stmts = []
                for u in update_exprs:
                    assert is_expression(u)
                    continue_block_stmts.append(node_factory.create_expression_stmt(u))
                continue_block_stmts.append(cont_node)
                block_stmt = node_factory.create_block_stmt(
                    node_factory.create_statement_list(continue_block_stmts)
                )

                for child_attr in cont_parent.get_children_names():
                    child = cont_parent.get_child_at(child_attr)
                    if child is not None and child is cont_node:
                        cont_parent.set_child_at(child_attr, block_stmt)

        # pack while loop
        while_body = self._wrap_loop_body(body_stmts)
        while_stmt = node_factory.create_while_stmt(condition, while_body)

        new_stmts.append(while_stmt)
        new_block = node_factory.create_block_stmt(
            node_factory.create_statement_list(new_stmts)
        )
        return (True, [new_block])
