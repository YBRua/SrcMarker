from .visitor import TransformingVisitor
from mutable_tree.nodes import Node, node_factory, LocalVariableDeclaration, StatementList, Identifier
from typing import Optional, List
from .var_same_type import split_DeclaratorList_by_Initializing


class MoveVarDeclToHeadVisitor(TransformingVisitor):

    def visit_StatementList(self,
                            node: StatementList,
                            parent: Optional[Node] = None,
                            parent_attr: Optional[str] = None):
        self.generic_visit(node, parent, parent_attr)
        new_children_list = []
        decl_list = []
        for child_attr in node.get_children_names():
            child = node.get_child_at(child_attr)
            if child is None:
                continue
            if isinstance(child, LocalVariableDeclaration):
                with_init_declarator_list, without_init_declarator_list = \
                    split_DeclaratorList_by_Initializing(child.declarators)
                if with_init_declarator_list is not None:
                    new_child = node_factory.create_local_variable_declaration(child.type, with_init_declarator_list)
                    new_children_list.append(new_child)
                if without_init_declarator_list is not None:
                    new_child = node_factory.create_local_variable_declaration(child.type, without_init_declarator_list)
                    decl_list.append(new_child)
            else:
                new_children_list.append(child)

        decl_list += new_children_list
        node.node_list = decl_list
        return False, []


class MoveVarDeclToBeforeUsedVisitor(TransformingVisitor):

    def visit_StatementList(self,
                            node: StatementList,
                            parent: Optional[Node] = None,
                            parent_attr: Optional[str] = None):
        new_children_list = []
        decl_map = {}
        for child_attr in node.get_children_names():
            child = node.get_child_at(child_attr)
            if child is None:
                continue
            if isinstance(child, LocalVariableDeclaration):
                with_init_declarator_list, without_init_declarator_list = \
                    split_DeclaratorList_by_Initializing(child.declarators)
                if with_init_declarator_list is not None:
                    new_child = node_factory.create_local_variable_declaration(child.type, with_init_declarator_list)
                    new_children_list.append(new_child)
                if without_init_declarator_list is not None:
                    for declarator in without_init_declarator_list.node_list:
                        declarator_list = node_factory.create_declarator_list([declarator])
                        new_child = node_factory.create_local_variable_declaration(child.type, declarator_list)
                        identifier = declarator.to_string()
                        decl_map[identifier] = new_child
            else:
                new_children_list.append(child)

        node.node_list = []
        for child in new_children_list:
            identifiers = []
            get_all_identifiers(child, identifiers)
            for identifier in identifiers:
                decl_node = decl_map.get(identifier, None)
                if decl_node is not None:
                    node.node_list.append(decl_node)
                    decl_map[identifier] = None
            node.node_list.append(child)

        # 一些变量声明但没有被使用到, 放到block最后
        for _, decl_node in decl_map.items():
            node.node_list.append(decl_node)

        self.generic_visit(node, parent, parent_attr)
        return False, []


def get_all_identifiers(node: Node, identifiers: List[str]):
    child_attrs = node.get_children_names()
    if len(child_attrs) == 0:
        if isinstance(node, Identifier):
            identifiers.append(node.to_string())
    else:
        for attr in child_attrs:
            child = node.get_child_at(attr)
            if child is None:
                continue
            get_all_identifiers(child, identifiers)
