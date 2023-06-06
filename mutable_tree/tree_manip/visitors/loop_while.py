from .visitor import TransformingVisitor
from mutable_tree.nodes import Node, NodeType
from mutable_tree.nodes import node_factory
from mutable_tree.nodes import ForStatement, Statement
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

    def visit_ForStatement(self,
                           node: ForStatement,
                           parent: Optional[Node] = None,
                           parent_attr: Optional[str] = None):
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
            condition = node_factory.create_literal('true')
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

        # pack while loop
        while_body = self._wrap_loop_body(body_stmts)
        while_stmt = node_factory.create_while_stmt(condition, while_body)

        new_stmts.append(while_stmt)
        return (True, new_stmts)
