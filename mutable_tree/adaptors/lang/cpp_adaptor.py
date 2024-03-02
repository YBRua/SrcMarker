# https://github.com/tree-sitter/tree-sitter-c/blob/master/grammar.js
# https://github.com/tree-sitter/tree-sitter-cpp/blob/master/grammar.js
import tree_sitter
from ...nodes import Expression, Statement
from ...nodes import (
    ArrayAccess,
    ArrayExpression,
    AssignmentExpression,
    BinaryExpression,
    CallExpression,
    CastExpression,
    FieldAccess,
    Identifier,
    Literal,
    NewExpression,
    TernaryExpression,
    ThisExpression,
    UnaryExpression,
    UpdateExpression,
    ParenthesizedExpression,
    ExpressionList,
    CommaExpression,
    SizeofExpression,
    PointerExpression,
    DeleteExpression,
    ScopeResolution,
    QualifiedIdentifier,
    CompoundLiteralExpression,
)
from ...nodes import (
    BlockStatement,
    BreakStatement,
    ContinueStatement,
    DoStatement,
    EmptyStatement,
    ExpressionStatement,
    ForInStatement,
    ForStatement,
    IfStatement,
    LabeledStatement,
    ReturnStatement,
    SwitchStatement,
    SwitchCase,
    SwitchCaseList,
    TryStatement,
    CatchClause,
    TryHandlers,
    WhileStatement,
    GotoStatement,
    ThrowStatement,
)
from ...nodes import (
    VariableDeclarator,
    ArrayDeclarator,
    PointerDeclarator,
    ReferenceDeclarator,
    InitializingDeclarator,
    LocalVariableDeclaration,
)
from ...nodes import (
    TypedFormalParameter,
    VariadicParameter,
    FormalParameterList,
    FunctionDeclarator,
    FunctionDeclaration,
)
from ...nodes import (
    TemplateDeclaration,
    TemplateParameterList,
    TypeParameterDeclaration,
)
from ...nodes import Modifier, ModifierList
from ...nodes import Program
from ...nodes import TypeIdentifier, Dimensions, DimensionSpecifier
from ...nodes import (
    get_assignment_op,
    get_binary_op,
    get_unary_op,
    get_update_op,
    get_field_access_op,
    get_pointer_op,
    get_typename_opts,
)
from ...nodes import node_factory
from typing import List


def convert_program(node: tree_sitter.Node) -> Program:
    assert node.type == "translation_unit", node.type
    stmts = []
    for child in node.children:
        stmts.append(convert_statement(child))
    return node_factory.create_program(node_factory.create_statement_list(stmts))


def convert_expression(node: tree_sitter.Node) -> Expression:
    expr_convertors = {
        "identifier": convert_identifier,
        "field_identifier": convert_identifier,
        "number_literal": convert_literal,
        "string_literal": convert_literal,
        "true": convert_literal,
        "false": convert_literal,
        "null": convert_literal,
        "nullptr": convert_literal,
        "concatenated_string": convert_literal,
        "char_literal": convert_literal,
        "raw_string_literal": convert_literal,
        "user_defined_literal": convert_literal,
        "assignment_expression": convert_assignment_expr,
        "binary_expression": convert_binary_expr,
        "update_expression": convert_update_expr,
        "unary_expression": convert_unary_expr,
        "cast_expression": convert_cast_expr,
        "conditional_expression": convert_ternary_expr,
        "subscript_expression": convert_array_access,
        "call_expression": convert_call_expr,
        "field_expression": convert_field_access,
        "parenthesized_expression": convert_parenthesized_expr,
        "comma_expression": convert_comma_expr,
        "sizeof_expression": convert_sizeof_expr,
        "pointer_expression": convert_pointer_expr,
        "this": convert_this_expr,
        "new_expression": convert_new_expr,
        "delete_expression": convert_delete_expr,
        "initializer_list": convert_array_expr,
        "qualified_identifier": convert_qualified_identifier,
        "compound_literal_expression": convert_compound_literal_expr,
        "template_function": convert_identifier,  # FIXME: replace with actual conversion
    }

    return expr_convertors[node.type](node)


def convert_statement(node: tree_sitter.Node) -> Statement:
    stmt_convertors = {
        "expression_statement": convert_expression_stmt,
        "labeled_statement": convert_labeled_stmt,
        "compound_statement": convert_block_stmt,
        "declaration": convert_local_variable_declaration,
        "for_statement": convert_for_stmt,
        "while_statement": convert_while_stmt,
        "break_statement": convert_break_stmt,
        "continue_statement": convert_continue_stmt,
        "do_statement": convert_do_stmt,
        "if_statement": convert_if_stmt,
        "for_range_loop": convert_for_range_loop,
        "labeled_statement": convert_labeled_stmt,
        "return_statement": convert_return_stmt,
        "switch_statement": convert_switch_stmt,
        "function_definition": convert_function_definition,
        "template_declaration": convert_template_declaration,
        "goto_statement": convert_goto_stmt,
        "throw_statement": convert_throw_stmt,
    }

    return stmt_convertors[node.type](node)


def convert_type(node: tree_sitter.Node) -> TypeIdentifier:
    type_convertors = {
        "type_descriptor": convert_simple_type,
        "type_identifier": convert_simple_type,
        "primitive_type": convert_simple_type,
        "placeholder_type_specifier": convert_simple_type,
        "sized_type_specifier": convert_simple_type,
        "template_type": convert_simple_type,
        "qualified_identifier": convert_qualified_type_identifier,
        "struct_specifier": convert_simple_type,  # FIXME: replace with actual conversion
    }

    return type_convertors[node.type](node)


def convert_modifier(node: tree_sitter.Node) -> Modifier:
    modifier = node.text.decode()
    return node_factory.create_modifier(modifier)


def convert_modifier_list(node: tree_sitter.Node) -> ModifierList:
    modifiers = []
    for child in node.children:
        modifiers.append(convert_modifier(child))
    return node_factory.create_modifier_list(modifiers)


def convert_simple_type(node: tree_sitter.Node) -> TypeIdentifier:
    return node_factory.create_type_identifier(node.text.decode())


def convert_identifier(node: tree_sitter.Node) -> Identifier:
    name = node.text.decode()
    return node_factory.create_identifier(name)


def convert_scope_resolution(node: tree_sitter.Node) -> ScopeResolution:
    scope_node = node.child_by_field_name("scope")
    if scope_node is None:
        scope = None
    else:
        scope = convert_identifier(scope_node)
    return node_factory.create_scope_resolution(scope)


def convert_qualified_identifier(node: tree_sitter.Node) -> QualifiedIdentifier:
    scope_resolution = convert_scope_resolution(node)
    name = convert_expression(node.child_by_field_name("name"))

    return node_factory.create_qualified_identifier(scope_resolution, name)


def convert_qualified_type_identifier(node: tree_sitter.Node) -> QualifiedIdentifier:
    scope_resolution = convert_scope_resolution(node)
    name = convert_type(node.child_by_field_name("name"))

    return node_factory.create_qualified_identifier(scope_resolution, name)


def convert_literal(node: tree_sitter.Node) -> Literal:
    value = node.text.decode()
    return node_factory.create_literal(value)


def convert_assignment_expr(node: tree_sitter.Node) -> AssignmentExpression:
    lhs = convert_expression(node.child_by_field_name("left"))
    rhs = convert_expression(node.child_by_field_name("right"))
    op = get_assignment_op(node.child_by_field_name("operator").text.decode())
    return node_factory.create_assignment_expr(lhs, rhs, op)


def convert_binary_expr(node: tree_sitter.Node) -> BinaryExpression:
    lhs = convert_expression(node.child_by_field_name("left"))
    rhs = convert_expression(node.child_by_field_name("right"))
    op = get_binary_op(node.child_by_field_name("operator").text.decode())
    return node_factory.create_binary_expr(lhs, rhs, op)


def convert_update_expr(node: tree_sitter.Node) -> UpdateExpression:
    if node.children[0].type not in {"++", "--"}:
        prefix = False
        op = get_update_op(node.children[1].text.decode())
        expr = convert_expression(node.children[0])
    else:
        prefix = True
        op = get_update_op(node.children[0].text.decode())
        expr = convert_expression(node.children[1])
    return node_factory.create_update_expr(expr, op, prefix)


def convert_unary_expr(node: tree_sitter.Node) -> UnaryExpression:
    op = get_unary_op(node.child_by_field_name("operator").text.decode())
    operand = convert_expression(node.child_by_field_name("argument"))
    return node_factory.create_unary_expr(operand, op)


def convert_cast_expr(node: tree_sitter.Node) -> CastExpression:
    # TODO: type intersections
    type_node = node.child_by_field_name("type")
    value_node = node.child_by_field_name("value")

    type_id = convert_type(type_node)
    value_expr = convert_expression(value_node)
    return node_factory.create_cast_expr(type_id, value_expr)


def convert_ternary_expr(node: tree_sitter.Node) -> TernaryExpression:
    condition_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    alternate_node = node.child_by_field_name("alternative")

    condition = convert_expression(condition_node)
    consequence = convert_expression(consequence_node)
    alternate = convert_expression(alternate_node)
    return node_factory.create_ternary_expr(condition, consequence, alternate)


def convert_array_access(node: tree_sitter.Node) -> ArrayAccess:
    array_expr = convert_expression(node.child_by_field_name("argument"))
    index_expr = convert_expression(node.child_by_field_name("index"))
    return node_factory.create_array_access(array_expr, index_expr)


def convert_argument_list(node: tree_sitter.Node) -> ExpressionList:
    args = []
    for ch in node.children[1:-1]:
        # skip parenthesis and comma sep
        if ch.type == ",":
            continue
        args.append(convert_expression(ch))
    return node_factory.create_expression_list(args)


def convert_call_expr(node: tree_sitter.Node) -> CallExpression:
    arg_node = node.child_by_field_name("arguments")
    args = convert_argument_list(arg_node)
    callee_node = node.child_by_field_name("function")
    if callee_node.type == "primitive_type":
        callee = convert_type(callee_node)
    else:
        callee = convert_expression(node.child_by_field_name("function"))
    return node_factory.create_call_expr(callee, args)


def convert_field_access(node: tree_sitter.Node) -> FieldAccess:
    if node.child_count != 3:
        raise RuntimeError(f"field access with {node.child_count} children")

    obj = convert_expression(node.child_by_field_name("argument"))
    name = convert_expression(node.child_by_field_name("field"))
    op = get_field_access_op(node.child_by_field_name("operator").text.decode())

    return node_factory.create_field_access(obj, name, op)


def convert_parenthesized_expr(node: tree_sitter.Node) -> ParenthesizedExpression:
    if node.child_count != 3:
        raise RuntimeError(f"parenthesized expr with {node.child_count} != 3 children")
    expr = convert_expression(node.children[1])
    return node_factory.create_parenthesized_expr(expr)


def convert_comma_expr(node: tree_sitter.Node) -> CommaExpression:
    left = convert_expression(node.child_by_field_name("left"))
    right = convert_expression(node.child_by_field_name("right"))
    return node_factory.create_comma_expr(left, right)


def convert_sizeof_expr(node: tree_sitter.Node) -> SizeofExpression:
    value_node = node.child_by_field_name("value")
    if value_node is not None:
        value = convert_expression(value_node)
        return node_factory.create_sizeof_expr(value)

    # must be a type
    type_node = node.child_by_field_name("type")
    type_id = convert_type(type_node)
    return node_factory.create_sizeof_expr(type_id)


def convert_pointer_expr(node: tree_sitter.Node) -> PointerExpression:
    expr = convert_expression(node.child_by_field_name("argument"))
    op = get_pointer_op(node.child_by_field_name("operator").text.decode())
    return node_factory.create_pointer_expr(expr, op)


def convert_this_expr(node: tree_sitter.Node) -> ThisExpression:
    return node_factory.create_this_expr()


def convert_array_expr(node: tree_sitter.Node) -> ArrayExpression:
    elements = []
    for ch in node.children[1:-1]:
        if ch.type == ",":
            continue
        elements.append(convert_expression(ch))
    return node_factory.create_array_expr(node_factory.create_expression_list(elements))


def convert_compound_literal_expr(node: tree_sitter.Node) -> CompoundLiteralExpression:
    type_id = convert_type(node.child_by_field_name("type"))
    value = convert_array_expr(node.child_by_field_name("value"))

    return node_factory.create_compound_literal_expr(type_id, value)


def convert_new_declarator(node: tree_sitter.Node) -> Dimensions:
    def _convert_child_declarator(node: tree_sitter.Node) -> List[DimensionSpecifier]:
        if node.child_count == 3:
            length = convert_expression(node.child_by_field_name("length"))
            return [node_factory.create_dimension_specifier(length)]
        elif node.child_count == 4:
            length = convert_expression(node.child_by_field_name("length"))
            return [
                node_factory.create_dimension_specifier(length)
            ] + _convert_child_declarator(node.children[-1])
        else:
            raise ValueError(f"invalid child count {node.child_count}")

    specifiers = _convert_child_declarator(node)
    return node_factory.create_dimensions(specifiers)


def convert_new_expr(node: tree_sitter.Node) -> NewExpression:
    placement_node = node.child_by_field_name("placement")
    if placement_node is not None:
        raise NotImplementedError("placement new")
    type_node = node.child_by_field_name("type")
    declarator_node = node.child_by_field_name("declarator")

    type_str = type_node.text.decode()
    if declarator_node is not None:
        new_declarator = convert_new_declarator(declarator_node)
    else:
        new_declarator = None

    type_id = node_factory.create_type_identifier(type_str, new_declarator)

    arguments_node = node.child_by_field_name("arguments")
    if arguments_node is not None:
        arguments = convert_argument_list(arguments_node)
    else:
        arguments = None

    return node_factory.create_new_expr(type_id, arguments)


def convert_delete_expr(node: tree_sitter.Node) -> DeleteExpression:
    expr = convert_expression(node.children[-1])
    is_array = node.children[-2].type == "]"
    return node_factory.create_delete_expr(expr, is_array)


def convert_expression_stmt(node: tree_sitter.Node) -> ExpressionStatement:
    if node.child_count == 1:
        # empty statement
        return node_factory.create_empty_stmt()
    expr = convert_expression(node.children[0])
    return node_factory.create_expression_stmt(expr)


def convert_block_stmt(node: tree_sitter.Node) -> BlockStatement:
    stmts = []
    for stmt_node in node.children[1:-1]:
        stmts.append(convert_statement(stmt_node))
    stmts = node_factory.create_statement_list(stmts)
    return node_factory.create_block_stmt(stmts)


def convert_condition_clause(node: tree_sitter.Node) -> Expression:
    initializer_node = node.child_by_field_name("initializer")
    if initializer_node is not None:
        raise NotImplementedError("condition clause with initializer")

    value_node = node.child_by_field_name("value")
    if value_node.type == "declaration":
        raise NotImplementedError("condition clause with declaration")

    return convert_expression(value_node)


def convert_if_stmt(node: tree_sitter.Node) -> IfStatement:
    cond_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    alternative_node = node.child_by_field_name("alternative")

    condition = convert_condition_clause(cond_node)
    consequence = convert_statement(consequence_node)
    if alternative_node is None:
        alternative = None
    else:
        alternative = convert_statement(alternative_node)

    return node_factory.create_if_stmt(condition, consequence, alternative)


def convert_do_stmt(node: tree_sitter.Node) -> DoStatement:
    body_node = node.child_by_field_name("body")
    cond_node = node.child_by_field_name("condition")

    assert cond_node.type == "parenthesized_expression"
    assert cond_node.child_count == 3
    cond_node = cond_node.children[1]

    body = convert_statement(body_node)
    cond = convert_expression(cond_node)
    return node_factory.create_do_stmt(cond, body)


def convert_goto_stmt(node: tree_sitter.Node) -> GotoStatement:
    target = convert_identifier(node.child_by_field_name("label"))
    return node_factory.create_goto_stmt(target)


def convert_variable_declarator(node: tree_sitter.Node) -> VariableDeclarator:
    return node_factory.create_variable_declarator(convert_identifier(node))


def convert_array_declarator(node: tree_sitter.Node) -> ArrayDeclarator:
    declarator = convert_declarator(node.child_by_field_name("declarator"))
    dim_node = node.child_by_field_name("size")
    if dim_node is not None:
        dim = convert_expression(dim_node)
    else:
        dim = None
    dim = node_factory.create_dimension_specifier(dim)
    return node_factory.create_array_declarator(declarator, dim)


def convert_pointer_declarator(node: tree_sitter.Node) -> PointerDeclarator:
    declarator = convert_declarator(node.child_by_field_name("declarator"))
    return node_factory.create_pointer_declarator(declarator)


def convert_init_declarator(node: tree_sitter.Node) -> InitializingDeclarator:
    declarator = convert_declarator(node.child_by_field_name("declarator"))

    value_node = node.child_by_field_name("value")
    if value_node.type == "argument_list":
        value = convert_argument_list(value_node)
    else:
        value = convert_expression(node.child_by_field_name("value"))
    return node_factory.create_initializing_declarator(declarator, value)


def convert_reference_declarator(node: tree_sitter.Node) -> ReferenceDeclarator:
    declarator = convert_declarator(node.children[1])
    r_ref = node.children[0].type == "&&"
    return node_factory.create_reference_declarator(declarator, r_ref)


def convert_function_declarator(node: tree_sitter.Node) -> FunctionDeclarator:
    declarator = convert_declarator(node.child_by_field_name("declarator"))
    params_node = node.child_by_field_name("parameters")
    if params_node is not None:
        params = convert_formal_parameters(params_node)
    else:
        params = None
    return node_factory.create_func_declarator(declarator, params)


def convert_declarator(node: tree_sitter.Node):
    decl_convertors = {
        "identifier": convert_variable_declarator,
        "pointer_declarator": convert_pointer_declarator,
        "array_declarator": convert_array_declarator,
        "init_declarator": convert_init_declarator,
        "reference_declarator": convert_reference_declarator,
        "function_declarator": convert_function_declarator,
        "operator_name": convert_variable_declarator,  # FIXME: replace with operator name
    }
    return decl_convertors[node.type](node)


def convert_formal_param(node: tree_sitter.Node) -> TypedFormalParameter:
    decl_type = _convert_declaration_specifiers(node)
    decl_node = node.child_by_field_name("declarator")
    # NOTE: cpp parser somehow treats variable declarations like std::vector v(size);
    # as function declarations, where size is a TypeIdentfier with no declarator node.
    if decl_node is not None:
        decl = convert_declarator(decl_node)
    else:
        decl = None

    return node_factory.create_typed_formal_param(decl_type, decl)


def convert_variadic_parameter(node: tree_sitter.Node) -> VariadicParameter:
    return node_factory.create_variadic_parameter()


def convert_formal_parameters(node: tree_sitter.Node) -> FormalParameterList:
    params = []
    for child in node.children[1:-1]:
        if child.type == ",":
            continue
        if child.type == "variadic_parameter":
            params.append(convert_variadic_parameter(child))
        else:
            # formal parameter
            params.append(convert_formal_param(child))
    return node_factory.create_formal_parameter_list(params)


def _convert_declaration_specifiers(node: tree_sitter.Node, start_idx: int = 0):
    def _is_decl_modifier(node: tree_sitter.Node) -> bool:
        modifier_types = {
            "storage_class_specifier",
            "type_qualifier",
            "attribute_specifier",
            "attribute_declaration",
            "ms_declspec_modifier",
        }
        return node.type in modifier_types

    prefix_specifier_nodes = []
    postfix_specifier_nodes = []

    idx = start_idx
    for child in node.children[idx:]:
        if not _is_decl_modifier(child):
            break
        idx += 1
        prefix_specifier_nodes.append(child)

    type_node = node.children[idx]

    idx += 1
    for child in node.children[idx:]:
        if not _is_decl_modifier(child):
            break
        postfix_specifier_nodes.append(child)

    if len(prefix_specifier_nodes):
        prefixes = [convert_modifier(n) for n in prefix_specifier_nodes]
        prefixes = node_factory.create_modifier_list(prefixes)
    else:
        prefixes = None
    if len(postfix_specifier_nodes):
        postfixes = [convert_modifier(n) for n in postfix_specifier_nodes]
        postfixes = node_factory.create_modifier_list(postfixes)
    else:
        postfixes = None
    ty = convert_type(type_node)

    return node_factory.create_declarator_type(ty, prefixes, postfixes)


def convert_local_variable_declaration(
    node: tree_sitter.Node,
) -> LocalVariableDeclaration:
    decl_type = _convert_declaration_specifiers(node)

    declarators = []
    for decl_node in node.children_by_field_name("declarator"):
        declarators.append(convert_declarator(decl_node))
    declarators = node_factory.create_declarator_list(declarators)

    return node_factory.create_local_variable_declaration(decl_type, declarators)


def convert_empty_stmt(node: tree_sitter.Node) -> EmptyStatement:
    return node_factory.create_empty_stmt()


def convert_for_stmt(node: tree_sitter.Node) -> ForStatement:
    init_nodes = node.children_by_field_name("initializer")
    cond_node = node.child_by_field_name("condition")
    update_node = node.children_by_field_name("update")
    body_node = node.child_by_field_name("body")

    body = convert_statement(body_node)
    if len(init_nodes) == 0:
        init = None
    else:
        if init_nodes[0].type == "declaration":
            assert len(init_nodes) == 1
            init = convert_local_variable_declaration(init_nodes[0])
        else:
            init = [convert_expression(init_node) for init_node in init_nodes]
            init = node_factory.create_expression_list(init)
    cond = convert_expression(cond_node) if cond_node is not None else None
    if len(update_node) == 0:
        update = None
    else:
        update = [convert_expression(update) for update in update_node]
        update = node_factory.create_expression_list(update)
    return node_factory.create_for_stmt(body, init, cond, update)


def convert_while_stmt(node: tree_sitter.Node) -> WhileStatement:
    cond_node = node.child_by_field_name("condition")
    assert cond_node.child_count == 3, "while condition with != 3 children"
    cond_node = cond_node.children[1]
    body_node = node.child_by_field_name("body")

    cond = convert_expression(cond_node)  # NOTE: should be condition clause
    body = convert_statement(body_node)
    return node_factory.create_while_stmt(cond, body)


def convert_break_stmt(node: tree_sitter.Node) -> BreakStatement:
    assert node.child_count == 2
    return node_factory.create_break_stmt()


def convert_continue_stmt(node: tree_sitter.Node) -> ContinueStatement:
    assert node.child_count == 2
    return node_factory.create_continue_stmt()


def convert_for_range_loop(node: tree_sitter.Node) -> ForInStatement:
    init_node = node.child_by_field_name("initializer")
    if init_node is not None:
        raise NotImplementedError("for range loop with initializer")

    decl_type = _convert_declaration_specifiers(node, start_idx=2)
    declarator_node = node.child_by_field_name("declarator")
    decl = convert_declarator(declarator_node)

    value = convert_expression(node.child_by_field_name("right"))
    body = convert_statement(node.child_by_field_name("body"))

    return node_factory.create_for_in_stmt(decl_type, decl, value, body)


def convert_labeled_stmt(node: tree_sitter.Node) -> LabeledStatement:
    assert node.child_count == 3
    label_node = node.children[0]
    stmt_node = node.children[2]

    label = convert_identifier(label_node)
    stmt = convert_statement(stmt_node)
    return node_factory.create_labeled_stmt(label, stmt)


def convert_return_stmt(node: tree_sitter.Node) -> ReturnStatement:
    if node.child_count == 2:
        expr_node = None
    else:
        assert node.child_count == 3
        expr_node = node.children[1]

    expr = convert_expression(expr_node) if expr_node is not None else None
    return node_factory.create_return_stmt(expr)


def convert_switch_case(node: tree_sitter.Node) -> SwitchCase:
    assert node.child_count >= 2

    if node.children[0].type == "default":
        start_idx = 2
        label = None
    else:
        assert node.children[0].type == "case"
        start_idx = 3
        label = convert_expression(node.child_by_field_name("value"))

    stmt_nodes = node.children[start_idx:]

    # convert body
    stmts = [convert_statement(stmt_node) for stmt_node in stmt_nodes]
    stmt_list = node_factory.create_statement_list(stmts)

    return node_factory.create_switch_case(stmt_list, label)


def convert_switch_block(node: tree_sitter.Node) -> SwitchCaseList:
    # ignore curly braces
    cases = []
    for child in node.children[1:-1]:
        cases.append(convert_switch_case(child))

    return node_factory.create_switch_case_list(cases)


def convert_switch_stmt(node: tree_sitter.Node) -> SwitchStatement:
    cond_node = node.child_by_field_name("condition")
    body_node = node.child_by_field_name("body")

    # NOTE: condition should be a condition clause
    assert cond_node.child_count == 3
    cond_node = cond_node.children[1]
    cond = convert_expression(cond_node)

    body = convert_switch_block(body_node)

    return node_factory.create_switch_stmt(cond, body)


def convert_catch_handler(node: tree_sitter.Node) -> CatchClause:
    raise NotImplementedError("catch handler")


def convert_try_handlers(handler_nodes: List[tree_sitter.Node]) -> TryHandlers:
    handlers = []
    for handler_node in handler_nodes:
        handlers.append(convert_catch_handler(handler_node))
    handlers = node_factory.create_try_handlers(handlers)

    return handlers


def convert_try_stmt(node: tree_sitter.Node) -> TryStatement:
    assert node.child_count >= 3

    # try body
    body_node = node.child_by_field_name("body")
    body = convert_block_stmt(body_node)

    # handlers and finalizer
    handler_nodes = node.children[2:]
    handlers = convert_try_handlers(handler_nodes)

    return node_factory.create_try_stmt(body, handlers)


def convert_throw_stmt(node: tree_sitter.Node) -> ThrowStatement:
    assert node.child_count == 3
    return node_factory.create_throw_stmt(convert_expression(node.children[1]))


def convert_function_definition(node: tree_sitter.Node) -> FunctionDeclaration:
    decl_type = _convert_declaration_specifiers(node, start_idx=0)
    decl = convert_declarator(node.child_by_field_name("declarator"))
    body = convert_block_stmt(node.child_by_field_name("body"))

    header = node_factory.create_func_header(decl, decl_type)
    return node_factory.create_func_declaration(header, body)


def convert_type_param_decl(node: tree_sitter.Node) -> TypeParameterDeclaration:
    assert node.child_count == 2
    typename_str = node.children[0].text.decode()
    typename_opt = get_typename_opts(typename_str)
    name_node = node.children[1]
    name = convert_type(name_node)
    return node_factory.create_type_parameter_declaration(name, typename_opt)


def convert_template_parameters(node: tree_sitter.Node) -> TemplateParameterList:
    # skip <>
    params = []
    for ch in node.children[1:-1]:
        if ch.type == ",":
            continue
        if ch.type == "parameter_declaration":
            params.append(convert_formal_param(ch))
        if ch.type == "type_parameter_declaration":
            params.append(convert_type_param_decl(ch))
    return node_factory.create_template_parameter_list(params)


def convert_template_declaration(node: tree_sitter.Node) -> TemplateDeclaration:
    params_node = node.child_by_field_name("parameters")
    params = convert_template_parameters(params_node)

    body_node = node.children[-1]
    body = convert_function_definition(body_node)

    return node_factory.create_template_declaration(params, body)
