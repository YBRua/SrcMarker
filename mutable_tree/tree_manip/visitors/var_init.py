from .visitor import TransformingVisitor
from mutable_tree.nodes import (Node, NodeType, node_factory, LocalVariableDeclaration,
                                StatementList, Declarator, InitializingDeclarator,
                                VariableDeclarator, AssignmentOps, ExpressionStatement,
                                AssignmentExpression, PointerDeclarator,
                                ReferenceDeclarator, ArrayDeclarator, FunctionDeclarator,
                                Identifier)
from typing import Optional, List
from .var_same_type import split_DeclaratorList_by_Initializing


class SplitVarInitAndDeclVisitor(TransformingVisitor):
    def visit_StatementList(self,
                            node: StatementList,
                            parent: Optional[Node] = None,
                            parent_attr: Optional[str] = None):
        self.generic_visit(node, parent, parent_attr)
        new_children_list = []
        for child_attr in node.get_children_names():
            child = node.get_child_at(child_attr)
            if child is None:
                continue
            if isinstance(child, LocalVariableDeclaration):
                with_init_declarator_list, without_init_declarator_list = \
                    split_DeclaratorList_by_Initializing(child.declarators)

                if without_init_declarator_list is not None:
                    declarators = without_init_declarator_list.node_list
                else:
                    declarators: List[Declarator] = []

                stmts = []
                if with_init_declarator_list is not None:
                    for initializing_declarator in with_init_declarator_list.node_list:

                        # dont split array expr initializer
                        if initializing_declarator.value.node_type == NodeType.ARRAY_EXPR:
                            declarators.append(initializing_declarator)
                            continue

                        variable_declarator, stmt = split_initializing_declarator(
                            initializing_declarator)
                        declarators.append(variable_declarator)
                        stmts.append(stmt)
                if len(declarators) > 0:
                    declarator_list = node_factory.create_declarator_list(declarators)
                    new_child = node_factory.create_local_variable_declaration(
                        child.type, declarator_list)
                    new_children_list.append(new_child)
                for stmt in stmts:
                    new_children_list.append(stmt)
            else:
                new_children_list.append(child)
        node.node_list = new_children_list
        return False, []


def split_initializing_declarator(node: InitializingDeclarator):
    identifier = get_identifier_from_declarator(node.declarator)
    variable_declarator = node_factory.create_variable_declarator(identifier)
    assignment_expression = node_factory.create_assignment_expr(
        identifier, node.value, AssignmentOps.EQUAL)
    stmt = node_factory.create_expression_stmt(assignment_expression)
    return variable_declarator, stmt


class MergeVarInitAndDeclVisitor(TransformingVisitor):
    def visit_StatementList(self,
                            node: StatementList,
                            parent: Optional[Node] = None,
                            parent_attr: Optional[str] = None):
        self.generic_visit(node, parent, parent_attr)
        # 第一次遍历找到所有只有声明没有init的变量, 及这些变量的第一个assignment
        # 第二次遍历用assignment给变量赋初值

        var_init = {}

        # 第一次遍历
        temp_children_list = []
        for child_attr in node.get_children_names():
            child = node.get_child_at(child_attr)
            if child is None:
                continue

            if isinstance(child, LocalVariableDeclaration):
                for declarator in child.declarators.node_list:
                    if not isinstance(declarator, InitializingDeclarator):
                        identifier_name = get_identifier_name_from_declarator(declarator)
                        var_init[identifier_name] = ('un_init', declarator)

            if is_assignment_used_in_init(child, var_init):
                # remove assignment
                pass
            else:
                temp_children_list.append(child)

        # 第二次遍历
        new_children_list = []
        for child in temp_children_list:
            if isinstance(child, LocalVariableDeclaration):
                declarator_list: List[Declarator] = []
                for declarator in child.declarators.node_list:
                    if not isinstance(declarator, InitializingDeclarator):
                        identifier_name = get_identifier_name_from_declarator(declarator)
                        init = var_init.get(identifier_name, None)
                        if init is not None and init[0] == 'inited':
                            declarator_list.append(init[1])
                        else:
                            declarator_list.append(declarator)
                    else:
                        declarator_list.append(declarator)
                declarators = node_factory.create_declarator_list(declarator_list)
                new_child = node_factory.create_local_variable_declaration(
                    child.type, declarators)
                new_children_list.append(new_child)
            else:
                new_children_list.append(child)

        node.node_list = new_children_list
        return False, []


def get_identifier_from_declarator(declarator: Declarator) -> Identifier:
    assert (isinstance(declarator, VariableDeclarator)
            or isinstance(declarator, PointerDeclarator)
            or isinstance(declarator, ReferenceDeclarator)
            or isinstance(declarator, ArrayDeclarator)
            or isinstance(declarator, FunctionDeclarator))
    if isinstance(declarator, VariableDeclarator):
        return declarator.decl_id
    else:
        declarator = declarator.declarator
        return get_identifier_from_declarator(declarator)


def get_identifier_name_from_declarator(declarator: Declarator) -> str:
    identifier = get_identifier_from_declarator(declarator)
    return identifier.name


def is_assignment_used_in_init(node: Node, var_init: {}) -> bool:
    if isinstance(node, ExpressionStatement) and isinstance(node.expr,
                                                            AssignmentExpression):
        assignment_node = node.expr
        left = assignment_node.left
        if isinstance(left, Identifier) and assignment_node.op == AssignmentOps.EQUAL:
            init = var_init.get(left.name, None)
            if init is not None and init[0] == 'un_init':
                init_declarator = node_factory.create_initializing_declarator(
                    init[1], assignment_node.right)
                var_init[left.name] = ('inited', init_declarator)
                return True
    return False
