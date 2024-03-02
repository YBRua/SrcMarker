from .visitor import TransformingVisitor
from mutable_tree.nodes import (
    Node,
    node_factory,
    LocalVariableDeclaration,
    ForStatement,
    StatementList,
    DeclaratorList,
    InitializingDeclarator,
    FunctionDeclarator,
    Declarator,
)
from mutable_tree.stringifiers import BaseStringifier
from typing import Optional, List


class SplitVarWithSameTypeVisitor(TransformingVisitor):
    def visit_LocalVariableDeclaration(
        self,
        node: LocalVariableDeclaration,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        node_list = node.declarators.node_list
        declarator_type = node.type
        if len(node_list) == 1:
            return False, []
        else:
            local_variable_declarations: List[Node] = []
            for declarator in node_list:
                new_declarator_list = node_factory.create_declarator_list([declarator])
                node = node_factory.create_local_variable_declaration(
                    declarator_type, new_declarator_list
                )
                local_variable_declarations.append(node)
            return True, local_variable_declarations

    def visit_ForStatement(
        self,
        node: ForStatement,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node.body, node, "body")
        new_node = node_factory.create_for_stmt(
            node.body, node.init, node.condition, node.update
        )
        return True, [new_node]


def split_DeclaratorList_by_Initializing(node: DeclaratorList):
    with_init_declarators: List[Declarator] = []
    without_init_declarators: List[Declarator] = []
    for declarator in node.node_list:
        if isinstance(declarator, InitializingDeclarator):
            with_init_declarators.append(declarator)
        elif isinstance(declarator, FunctionDeclarator):
            # special case, sometimes tree-sitter recognizes a initializer as function declarator
            with_init_declarators.append(declarator)
        else:
            without_init_declarators.append(declarator)

    if len(with_init_declarators) == 0:
        with_init_declarator_list = None
    else:
        with_init_declarator_list = node_factory.create_declarator_list(
            with_init_declarators
        )

    if len(without_init_declarators) == 0:
        without_init_declarator_list = None
    else:
        without_init_declarator_list = node_factory.create_declarator_list(
            without_init_declarators
        )

    return with_init_declarator_list, without_init_declarator_list


class MergeVarWithSameTypeVisitor(TransformingVisitor):
    def visit_StatementList(
        self,
        node: StatementList,
        parent: Optional[Node] = None,
        parent_attr: Optional[str] = None,
    ):
        self.generic_visit(node, parent, parent_attr)
        new_children_list = []
        types = {}
        type_loc_map = {}

        stringifier = BaseStringifier()
        for child_attr in node.get_children_names():
            child = node.get_child_at(child_attr)
            if child is None:
                continue
            if isinstance(child, LocalVariableDeclaration):
                (
                    with_init_declarator_list,
                    without_init_declarator_list,
                ) = split_DeclaratorList_by_Initializing(child.declarators)
                if with_init_declarator_list is not None:
                    new_child = node_factory.create_local_variable_declaration(
                        child.type, with_init_declarator_list
                    )
                    new_children_list.append(new_child)

                if without_init_declarator_list is not None:
                    child_type_str = stringifier.stringify(child.type.type_id)
                    if types.get(child_type_str, None) is None:
                        types[child_type_str] = (
                            child.type,
                            without_init_declarator_list,
                        )
                        # register the location of the first declaration of this type
                        # NOTE: unsafe. should sort variables by their order of declaration
                        new_children_list.append(
                            node_factory.create_local_variable_declaration(
                                child.type, without_init_declarator_list
                            )
                        )
                        type_loc_map[child_type_str] = len(new_children_list) - 1
                    else:
                        # merge all the same type variables to the first one
                        decl_node = new_children_list[type_loc_map[child_type_str]]
                        assert isinstance(decl_node, LocalVariableDeclaration)
                        decl_node.declarators.node_list.extend(
                            without_init_declarator_list.node_list
                        )
            else:
                new_children_list.append(child)

        # for child_type_str, (child_type, decl_list) in types.items():
        #     new_child = node_factory.create_local_variable_declaration(
        #         child_type, decl_list)
        #     new_children_list.insert(0, new_child)
        node.node_list = new_children_list
        return False, []
