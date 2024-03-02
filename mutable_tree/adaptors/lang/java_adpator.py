# https://github.com/tree-sitter/tree-sitter-java/blob/master/grammar.js
import tree_sitter
from ...nodes import Expression, Statement
from ...nodes import (
    ArrayAccess,
    ArrayCreationExpression,
    ArrayExpression,
    AssignmentExpression,
    BinaryExpression,
    CallExpression,
    CastExpression,
    FieldAccess,
    Identifier,
    InstanceofExpression,
    Literal,
    NewExpression,
    TernaryExpression,
    ThisExpression,
    UnaryExpression,
    UpdateExpression,
    ParenthesizedExpression,
    ExpressionList,
    LambdaExpression,
)
from ...nodes import (
    AssertStatement,
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
    ThrowStatement,
    TryStatement,
    CatchClause,
    FinallyClause,
    TryHandlers,
    WhileStatement,
    YieldStatement,
    TryResource,
    TryResourceList,
    TryWithResourcesStatement,
    SynchronizedStatement,
)
from ...nodes import (
    Declarator,
    VariableDeclarator,
    ArrayDeclarator,
    InitializingDeclarator,
    DeclaratorType,
    LocalVariableDeclaration,
)
from ...nodes import (
    FormalParameter,
    UntypedParameter,
    TypedFormalParameter,
    SpreadParameter,
    FormalParameterList,
    FunctionHeader,
    FunctionDeclaration,
)
from ...nodes import Modifier, ModifierList
from ...nodes import Program
from ...nodes import TypeIdentifier, Dimensions, TypeParameter, TypeParameterList
from ...nodes import get_assignment_op, get_binary_op, get_unary_op, get_update_op
from ...nodes import node_factory
from typing import Tuple, Optional, List


def convert_program(node: tree_sitter.Node) -> Program:
    assert node.type == "program"
    stmts = []
    for ch in node.children:
        stmts.append(convert_statement(ch))
    return node_factory.create_program(node_factory.create_statement_list(stmts))


def convert_expression(node: tree_sitter.Node) -> Expression:
    expr_convertors = {
        "identifier": convert_identifier,
        # FIXME: values for literals are currently always strings
        "decimal_integer_literal": convert_literal,
        "hex_integer_literal": convert_literal,
        "octal_integer_literal": convert_literal,
        "binary_integer_literal": convert_literal,
        "decimal_floating_point_literal": convert_literal,
        "hex_floating_point_literal": convert_literal,
        "true": convert_literal,
        "false": convert_literal,
        "character_literal": convert_literal,
        "string_literal": convert_literal,
        "null_literal": convert_literal,
        # NOTE: things below are not literals in the grammar,
        # but we temporarily ignore the differences here
        "class_literal": convert_literal,
        "super": convert_literal,
        "array_access": convert_array_access,
        "assignment_expression": convert_assignment_expr,
        "binary_expression": convert_binary_expr,
        "method_invocation": convert_call_expr,
        "field_access": convert_field_access,
        "update_expression": convert_update_expr,
        "object_creation_expression": convert_new_expr,
        "cast_expression": convert_cast_expr,
        "instanceof_expression": convert_instanceof_expr,
        "ternary_expression": convert_ternary_expr,
        "this": convert_this_expr,
        "unary_expression": convert_unary_expr,
        "parenthesized_expression": convert_parenthesized_expr,
        "condition": convert_parenthesized_expr,  # they seem to be the same
        "array_creation_expression": convert_array_creation_expr,
        # NOTE: array_initializer is not an expression, but we treat it as one
        "array_initializer": convert_array_expr,
        "lambda_expression": convert_lambda_expr,
        "switch_expression": convert_switch_stmt,
    }

    return expr_convertors[node.type](node)


def convert_statement(node: tree_sitter.Node) -> Statement:
    stmt_convertors = {
        ";": convert_empty_stmt,
        "local_variable_declaration": convert_local_variable_declaration,
        "expression_statement": convert_expression_stmt,
        "empty_statement": convert_empty_stmt,
        "block": convert_block_stmt,
        "for_statement": convert_for_stmt,
        "while_statement": convert_while_stmt,
        "assert_statement": convert_assert_stmt,
        "break_statement": convert_break_stmt,
        "continue_statement": convert_continue_stmt,
        "do_statement": convert_do_stmt,
        "enhanced_for_statement": convert_enhanced_for_stmt,
        "if_statement": convert_if_stmt,
        "labeled_statement": convert_labeled_stmt,
        "return_statement": convert_return_stmt,
        "switch_expression": convert_switch_stmt,
        "try_statement": convert_try_stmt,
        "try_with_resources_statement": convert_try_with_resources_stmt,
        "throw_statement": convert_throw_stmt,
        "yield_statement": convert_yield_stmt,
        "synchronized_statement": convert_synchronized_stmt,
    }

    return stmt_convertors[node.type](node)


def convert_type(node: tree_sitter.Node) -> TypeIdentifier:
    type_convertors = {
        "array_type": convert_array_type,
        "void_type": convert_simple_type,
        "integral_type": convert_simple_type,
        "floating_point_type": convert_simple_type,
        "boolean_type": convert_simple_type,
        "scoped_type_identifier": convert_simple_type,
        "generic_type": convert_simple_type,
        "type_identifier": convert_simple_type,
        "identifier": convert_simple_type,
    }

    return type_convertors[node.type](node)


def convert_identifier(node: tree_sitter.Node) -> Identifier:
    name = node.text.decode()
    return node_factory.create_identifier(name)


def convert_literal(node: tree_sitter.Node) -> Literal:
    value = node.text.decode()
    return node_factory.create_literal(value)


def convert_array_access(node: tree_sitter.Node) -> ArrayAccess:
    array_expr = convert_expression(node.child_by_field_name("array"))
    index_expr = convert_expression(node.child_by_field_name("index"))
    return node_factory.create_array_access(array_expr, index_expr)


def convert_array_expr(node: tree_sitter.Node) -> ArrayExpression:
    elements = []
    for ch in node.children[1:-1]:
        if ch.type == ",":
            continue
        elements.append(convert_expression(ch))
    return node_factory.create_array_expr(node_factory.create_expression_list(elements))


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


def convert_cast_expr(node: tree_sitter.Node) -> CastExpression:
    # TODO: type intersections
    type_node = node.child_by_field_name("type")
    value_node = node.child_by_field_name("value")

    type_id = convert_type(type_node)
    value_expr = convert_expression(value_node)
    return node_factory.create_cast_expr(type_id, value_expr)


def convert_field_access(node: tree_sitter.Node) -> FieldAccess:
    if node.child_count != 3:
        raise RuntimeError(f"field access with {node.child_count} children")

    obj_expr = convert_expression(node.child_by_field_name("object"))
    name_expr = convert_expression(node.child_by_field_name("field"))
    return node_factory.create_field_access(obj_expr, name_expr)


def convert_argument_list(node: tree_sitter.Node) -> ExpressionList:
    args = []
    for ch in node.children[1:-1]:
        # skip parenthesis and comma sep
        if ch.type == ",":
            continue
        args.append(convert_expression(ch))
    return node_factory.create_expression_list(args)


def convert_call_expr(node: tree_sitter.Node) -> CallExpression:
    if node.child_count != 2 and node.child_count != 4:
        raise RuntimeError(f"call expr with {node.child_count} children")

    arg_node = node.child_by_field_name("arguments")
    args = convert_argument_list(arg_node)

    name_expr = convert_expression(node.child_by_field_name("name"))
    obj_node = node.child_by_field_name("object")
    if obj_node is not None:
        obj_expr = convert_expression(obj_node)
        callee_expr = node_factory.create_field_access(obj_expr, name_expr)
    else:
        callee_expr = name_expr

    return node_factory.create_call_expr(callee_expr, args)


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


def convert_new_expr(node: tree_sitter.Node) -> NewExpression:
    # TODO: type_argument, class body
    # type_arg_node = node.child_by_field_name('type_arguments')
    type_node = node.child_by_field_name("type")
    args_node = node.child_by_field_name("arguments")

    type_id = convert_type(type_node)
    args = convert_argument_list(args_node)

    if node.children[-1].type == "class_body":
        raise NotImplementedError("class_body in object creation is not supported")

    return node_factory.create_new_expr(type_id, args)


def convert_array_creation_expr(node: tree_sitter.Node) -> ArrayCreationExpression:
    type_node = node.child_by_field_name("type")
    type_id = convert_type(type_node)

    dim_nodes = node.children_by_field_name("dimensions")
    dim_specifiers = []
    for dim_node in dim_nodes:
        if dim_node.type == "dimensions_expr":
            if dim_node.child_count != 3:
                raise NotImplementedError("dimensions_expr with annotations")
            expr = convert_expression(dim_node.children[1])
            dim_specifiers.append(node_factory.create_dimension_specifier(expr))
        else:
            n_dims = dim_node.text.decode().count("[")
            for _ in range(n_dims):
                dim_specifiers.append(node_factory.create_dimension_specifier())
    dims = node_factory.create_dimensions(dim_specifiers)

    value_node = node.child_by_field_name("value")
    value = convert_array_expr(value_node) if value_node is not None else None

    return node_factory.create_array_creation_expr(type_id, dims, value)


def convert_instanceof_expr(node: tree_sitter.Node) -> InstanceofExpression:
    # TODO: name field, final modifier (children[2])
    # name_node = node.child_by_field_name('name')
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")

    left = convert_expression(left_node)
    right = convert_type(right_node)
    return node_factory.create_instanceof_expr(left, right)


def convert_ternary_expr(node: tree_sitter.Node) -> TernaryExpression:
    condition_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    alternate_node = node.child_by_field_name("alternative")

    condition = convert_expression(condition_node)
    consequence = convert_expression(consequence_node)
    alternate = convert_expression(alternate_node)
    return node_factory.create_ternary_expr(condition, consequence, alternate)


def convert_this_expr(node: tree_sitter.Node) -> ThisExpression:
    return node_factory.create_this_expr()


def convert_unary_expr(node: tree_sitter.Node) -> UnaryExpression:
    op = get_unary_op(node.child_by_field_name("operator").text.decode())
    operand = convert_expression(node.child_by_field_name("operand"))
    return node_factory.create_unary_expr(operand, op)


def convert_parenthesized_expr(node: tree_sitter.Node) -> ParenthesizedExpression:
    assert node.child_count == 3, "parenthesized expr with != 3 children"
    expr = convert_expression(node.children[1])
    return node_factory.create_parenthesized_expr(expr)


def convert_lambda_expr(node: tree_sitter.Node) -> LambdaExpression:
    # params
    param_node = node.child_by_field_name("parameters")

    parenthesized = False
    if param_node.type == "identifier":
        param = convert_identifier(param_node)
        param = node_factory.create_variable_declarator(param)
        param = node_factory.create_untyped_param(param)
        params = node_factory.create_formal_parameter_list([param])
    elif param_node.type == "formal_parameters":
        parenthesized = True
        params = convert_formal_parameters(param_node)
    elif param_node.type == "inferred_parameters":
        parenthesized = True
        params = []
        for ch in param_node.children[1:-1]:
            if ch.type == ",":
                continue
            param = convert_identifier(ch)
            param = node_factory.create_variable_declarator(param)
            params.append(node_factory.create_untyped_param(param))
        params = node_factory.create_formal_parameter_list(params)

    # body
    body_node = node.child_by_field_name("body")
    if body_node.type == "block":
        body = convert_block_stmt(body_node)
    else:
        body = convert_expression(body_node)

    return node_factory.create_lambda_expr(params, body, parenthesized)


def convert_dimensions(node: tree_sitter.Node) -> Dimensions:
    n_dims = node.text.decode().count("[")
    dims = []
    for _ in range(n_dims):
        dims.append(node_factory.create_dimension_specifier())
    return node_factory.create_dimensions(dims)


def convert_simple_type(node: tree_sitter.Node) -> TypeIdentifier:
    return node_factory.create_type_identifier(node.text.decode())


def convert_array_type(node: tree_sitter.Node) -> TypeIdentifier:
    element_ty = convert_type(node.child_by_field_name("element"))
    dimensions = convert_dimensions(node.child_by_field_name("dimensions"))
    return node_factory.create_array_type(element_ty, dimensions)


def convert_expression_stmt(node: tree_sitter.Node) -> ExpressionStatement:
    expr = convert_expression(node.children[0])
    return node_factory.create_expression_stmt(expr)


def convert_variable_declarator_id(node: tree_sitter.Node) -> Declarator:
    """Performs conversion on _variable_declarator_id.

    Returns a VariableDeclarator or an ArrayDeclarator
    """
    name_node = node.child_by_field_name("name")
    name = convert_identifier(name_node)
    decl = node_factory.create_variable_declarator(name)

    dim_node = node.child_by_field_name("dimensions")
    if dim_node is not None:
        n_dims = dim_node.text.decode().count("[")
        for _ in range(n_dims):
            dim = node_factory.create_dimension_specifier()
            decl = node_factory.create_array_declarator(decl, dim)
    return decl


def convert_variable_declarator(node: tree_sitter.Node) -> Declarator:
    value_node = node.child_by_field_name("value")
    value = convert_expression(value_node) if value_node is not None else None

    decl = convert_variable_declarator_id(node)
    if value is not None:
        decl = node_factory.create_initializing_declarator(decl, value)

    return decl


def convert_local_variable_declaration(
    node: tree_sitter.Node,
) -> LocalVariableDeclaration:
    if node.children[0].type == "modifiers":
        modifiers = convert_modifier_list(node.children[0])
    else:
        modifiers = None

    ty = convert_type(node.child_by_field_name("type"))
    decl_type = node_factory.create_declarator_type(ty, prefixes=modifiers)

    declarators = []
    for decl_node in node.children_by_field_name("declarator"):
        declarators.append(convert_variable_declarator(decl_node))
    declarators = node_factory.create_declarator_list(declarators)

    return node_factory.create_local_variable_declaration(decl_type, declarators)


def convert_empty_stmt(node: tree_sitter.Node) -> EmptyStatement:
    return node_factory.create_empty_stmt()


def convert_block_stmt(node: tree_sitter.Node) -> BlockStatement:
    stmts = []
    for stmt_node in node.children[1:-1]:
        stmts.append(convert_statement(stmt_node))
    stmts = node_factory.create_statement_list(stmts)
    return node_factory.create_block_stmt(stmts)


def convert_for_stmt(node: tree_sitter.Node) -> ForStatement:
    init_nodes = node.children_by_field_name("init")
    cond_node = node.child_by_field_name("condition")
    update_node = node.children_by_field_name("update")
    body_node = node.child_by_field_name("body")

    body = convert_statement(body_node)
    if len(init_nodes) == 0:
        init = None
    else:
        if init_nodes[0].type == "local_variable_declaration":
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

    cond = convert_expression(cond_node)
    body = convert_statement(body_node)
    return node_factory.create_while_stmt(cond, body)


def convert_assert_stmt(node: tree_sitter.Node) -> AssertStatement:
    if node.child_count == 3:
        cond_node = node.children[1]
        msg_node = None
    else:
        assert node.child_count == 5
        cond_node = node.children[1]
        msg_node = node.children[3]

    cond = convert_expression(cond_node)
    msg = convert_expression(msg_node) if msg_node is not None else None
    return node_factory.create_assert_stmt(cond, msg)


def convert_break_stmt(node: tree_sitter.Node) -> BreakStatement:
    if node.child_count == 2:
        label_node = None
    else:
        assert node.child_count == 3
        label_node = node.children[1]

    label = convert_expression(label_node) if label_node is not None else None
    return node_factory.create_break_stmt(label)


def convert_continue_stmt(node: tree_sitter.Node) -> ContinueStatement:
    if node.child_count == 2:
        label_node = None
    else:
        assert node.child_count == 3
        label_node = node.children[1]

    label = convert_expression(label_node) if label_node is not None else None
    return node_factory.create_continue_stmt(label)


def convert_do_stmt(node: tree_sitter.Node) -> DoStatement:
    body_node = node.child_by_field_name("body")
    cond_node = node.child_by_field_name("condition")

    assert cond_node.type == "parenthesized_expression"
    assert cond_node.child_count == 3
    cond_node = cond_node.children[1]

    body = convert_statement(body_node)
    cond = convert_expression(cond_node)
    return node_factory.create_do_stmt(cond, body)


def convert_enhanced_for_stmt(node: tree_sitter.Node) -> ForInStatement:
    if node.child_count == 9:
        modifiers = convert_modifier_list(node.children[2])
        offset = 1
    else:
        modifiers = None
        offset = 0

    # without modifiers
    type_node = node.child_by_field_name("type")
    var_decl_node = node.children[3 + offset]
    value_node = node.child_by_field_name("value")
    body_node = node.child_by_field_name("body")

    ty = convert_type(type_node)
    decl_ty = node_factory.create_declarator_type(ty, modifiers)
    # NOTE: should be variable declarator according to grammar specification
    var_decl = convert_identifier(var_decl_node)
    var_decl = node_factory.create_variable_declarator(var_decl)

    value = convert_expression(value_node)
    body = convert_statement(body_node)
    return node_factory.create_for_in_stmt(decl_ty, var_decl, value, body)


def convert_if_stmt(node: tree_sitter.Node) -> IfStatement:
    cond_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    alternative_node = node.child_by_field_name("alternative")

    assert cond_node.type == "condition" and cond_node.child_count == 3
    cond_node = cond_node.children[1]

    condition = convert_expression(cond_node)
    consequence = convert_statement(consequence_node)
    if alternative_node is None:
        alternative = None
    else:
        alternative = convert_statement(alternative_node)

    return node_factory.create_if_stmt(condition, consequence, alternative)


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
    label_node = node.children[0]
    stmt_nodes = node.children[2:]

    # convert label
    if label_node.child_count == 1:
        # default
        label = None
    else:
        assert label_node.child_count == 2
        label = convert_expression(label_node.children[1])

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

    assert cond_node.type == "parenthesized_expression" and cond_node.child_count == 3
    cond_node = cond_node.children[1]

    cond = convert_expression(cond_node)
    body = convert_switch_block(body_node)

    return node_factory.create_switch_stmt(cond, body)


def convert_throw_stmt(node: tree_sitter.Node) -> ThrowStatement:
    assert node.child_count == 3
    expr_node = node.children[1]

    expr = convert_expression(expr_node)
    return node_factory.create_throw_stmt(expr)


def convert_catch_handler(node: tree_sitter.Node) -> CatchClause:
    assert node.child_count == 5
    param_node = node.children[2]
    body_node = node.child_by_field_name("body")
    body = convert_block_stmt(body_node)

    # catch param list
    if param_node.child_count == 3:
        modifiers = convert_modifier_list(param_node.children[0])
        catch_type_node = param_node.children[1]
        decl_node = param_node.children[2]
    else:
        assert param_node.child_count == 2
        modifiers = None
        catch_type_node = param_node.children[0]
        decl_node = param_node.children[1]

    param_types = []
    for ty in catch_type_node.children:
        # skip sep
        if ty.type == "|":
            continue
        param_types.append(convert_type(ty))
    param_types = node_factory.create_type_identifier_list(param_types)

    # NOTE: should be variable declarator according to grammar specification
    decl = convert_identifier(decl_node)

    return node_factory.create_catch_clause(body, param_types, decl, modifiers)


def convert_try_finalizer(node: tree_sitter.Node) -> FinallyClause:
    assert node.child_count == 2
    body_node = node.children[1]
    body = convert_block_stmt(body_node)
    return node_factory.create_finally_clause(body)


def convert_try_handlers(
    handler_nodes: List[tree_sitter.Node],
) -> Tuple[Optional[TryHandlers], Optional[FinallyClause]]:
    if len(handler_nodes) == 0:
        # try-with-resources allow empty catch and finally
        return None, None

    if handler_nodes[-1].type == "finally_clause":
        finally_node = handler_nodes[-1]
        handler_nodes = handler_nodes[:-1]
        finalizer = convert_try_finalizer(finally_node)
    else:
        finalizer = None
    handlers = []
    for handler_node in handler_nodes:
        handlers.append(convert_catch_handler(handler_node))
    handlers = node_factory.create_try_handlers(handlers)

    return handlers, finalizer


def convert_try_stmt(node: tree_sitter.Node) -> TryStatement:
    assert node.child_count >= 3

    # try body
    body_node = node.child_by_field_name("body")
    body = convert_block_stmt(body_node)

    # handlers and finalizer
    handler_nodes = node.children[2:]
    handlers, finalizer = convert_try_handlers(handler_nodes)

    return node_factory.create_try_stmt(body, handlers, finalizer)


def convert_resource(node: tree_sitter.Node) -> TryResource:
    if node.child_count == 1:
        # identifier or field access
        assert node.children[0].type in {"identifier", "field_access"}
        resource = convert_expression(node.children[0])
    else:
        if node.child_count != 4:
            modifier_nodes = node.children[:-4]
        else:
            modifier_nodes = None
        if modifier_nodes is not None:
            modifiers = [convert_modifier(n) for n in modifier_nodes]
            modifiers = node_factory.create_modifier_list(modifiers)
        else:
            modifiers = None

        ty = convert_type(node.child_by_field_name("type"))
        decl_type = node_factory.create_declarator_type(ty, modifiers)

        declarator = convert_variable_declarator_id(node)
        value = convert_expression(node.child_by_field_name("value"))
        declarator = node_factory.create_initializing_declarator(declarator, value)
        declarators = node_factory.create_declarator_list([declarator])

        resource = node_factory.create_local_variable_declaration(
            decl_type, declarators
        )

    return node_factory.create_try_resource(resource)


def convert_resource_specification(node: tree_sitter.Node) -> TryResourceList:
    resources = []
    # skip parentheses
    for resource in node.children[1:-1]:
        if resource.type == ";":
            continue
        resources.append(convert_resource(resource))
    resources = node_factory.create_try_resource_list(resources)
    return resources


def convert_try_with_resources_stmt(
    node: tree_sitter.Node,
) -> TryWithResourcesStatement:
    assert node.child_count >= 3, node.child_count

    # resources
    resource_nodes = node.child_by_field_name("resources")
    resources = convert_resource_specification(resource_nodes)

    # body
    body_node = node.child_by_field_name("body")
    body = convert_block_stmt(body_node)

    # handlers and finalizer
    handler_nodes = node.children[3:]
    handlers, finalizer = convert_try_handlers(handler_nodes)

    return node_factory.create_try_with_resources_stmt(
        resources, body, handlers, finalizer
    )


def convert_yield_stmt(node: tree_sitter.Node) -> YieldStatement:
    assert node.child_count == 3
    expr_node = node.children[1]

    expr = convert_expression(expr_node)
    return node_factory.create_yield_stmt(expr)


def convert_synchronized_stmt(node: tree_sitter.Node) -> SynchronizedStatement:
    assert node.child_count == 3, node.child_count
    expr_node = node.children[1]
    body_node = node.child_by_field_name("body")

    expr = convert_parenthesized_expr(expr_node)
    body = convert_block_stmt(body_node)
    return node_factory.create_synchronized_stmt(expr, body)


def convert_modifier(node: tree_sitter.Node) -> Modifier:
    modifier = node.text.decode()
    return node_factory.create_modifier(modifier)


def convert_modifier_list(node: tree_sitter.Node) -> ModifierList:
    modifiers = []
    for child in node.children:
        modifiers.append(convert_modifier(child))
    return node_factory.create_modifier_list(modifiers)


def convert_formal_param(node: tree_sitter.Node) -> TypedFormalParameter:
    assert node.type == "formal_parameter", node.type
    if node.children[0].type == "modifiers":
        modifiers_node = node.children[0]
        modifiers = convert_modifier_list(modifiers_node)
    else:
        modifiers = None

    type_node = node.child_by_field_name("type")
    type_id = convert_type(type_node)
    decl_type = node_factory.create_declarator_type(type_id, modifiers)
    decl = convert_variable_declarator_id(node)

    return node_factory.create_typed_formal_param(decl_type, decl)


def convert_spread_param(node: tree_sitter.Node) -> SpreadParameter:
    assert node.type == "spread_parameter", node.type
    idx = 0
    if node.children[idx].type == "modifiers":
        modifiers_node = node.children[idx]
        modifiers = convert_modifier_list(modifiers_node)
        idx += 1
    else:
        modifiers = None

    type_node = node.children[idx]
    type_id = convert_type(type_node)
    decl_type = node_factory.create_declarator_type(type_id, modifiers)
    idx += 2  # skip '...'

    # spread parameter uses variable declarator, which is a independent node
    decl = convert_variable_declarator_id(node.children[idx])

    return node_factory.create_spread_param(decl, decl_type)


def convert_formal_parameters(node: tree_sitter.Node) -> FormalParameterList:
    params = []
    for child in node.children[1:-1]:
        if child.type == ",":
            continue
        if child.type == "spread_parameter":
            params.append(convert_spread_param(child))
        else:
            # formal parameter
            params.append(convert_formal_param(child))
    return node_factory.create_formal_parameter_list(params)


def convert_type_param(node: tree_sitter.Node) -> TypeParameter:
    if (
        node.children[0].type != "identifier"
        and node.children[0].type != "type_identifier"
    ):
        raise NotImplementedError("type parameter with annotataions")

    # name
    name = node.children[0].text.decode()

    # bounds
    bounds = None
    if node.children[-1].type == "type_bound":
        bounds = []
        bound_node = node.children[-1]
        for child in bound_node.children[1:]:
            if child.type == "&":
                continue
            bounds.append(convert_type(child))
        bounds = node_factory.create_type_identifier_list(bounds)

    return node_factory.create_type_parameter(name, bounds)


def convert_type_params(node: tree_sitter.Node) -> TypeParameterList:
    params = []
    for child in node.children[1:-1]:
        if child.type == ",":
            continue
        params.append(convert_type_param(child))
    return node_factory.create_type_parameter_list(params)


def convert_function_header(node: tree_sitter.Node) -> FunctionHeader:
    # NOTE: the node should be method_declaration
    assert node.type == "method_declaration", node.type

    # modifiers
    idx = 0
    if node.children[idx].type == "modifiers":
        modifiers_node = node.children[idx]
        modifiers = convert_modifier_list(modifiers_node)
        idx += 1
    else:
        modifiers = None

    # type parameter list
    type_params_node = node.child_by_field_name("type_parameters")
    if type_params_node is not None:
        type_params = convert_type_params(type_params_node)
    else:
        type_params = None

    # header
    type_node = node.child_by_field_name("type")
    type_id = convert_type(type_node)
    return_type = node_factory.create_declarator_type(type_id)
    name_node = node.child_by_field_name("name")
    name = convert_identifier(name_node)
    params_node = node.child_by_field_name("parameters")
    params = convert_formal_parameters(params_node)
    dim_node = node.child_by_field_name("dimensions")
    if dim_node is not None:
        dim = convert_dimensions(dim_node)
    else:
        dim = None

    # FIXME: maybe optimize this
    throws = None
    for ch in node.children:
        if ch.type == "throws":
            throws = []
            # ignore 'throws'
            for child in ch.children[1:]:
                if child.type == ",":
                    continue
                throws.append(convert_type(child))
            throws = node_factory.create_type_identifier_list(throws)

    name = node_factory.create_variable_declarator(name)
    func_decl = node_factory.create_func_declarator(name, params)

    return node_factory.create_func_header(
        func_decl, return_type, dim, throws, modifiers, type_params
    )


def convert_function_declaration(node: tree_sitter.Node) -> FunctionDeclaration:
    declarator = convert_function_header(node)
    body_node = node.child_by_field_name("body")
    if body_node is not None:
        body = convert_block_stmt(body_node)
    else:
        body = node_factory.create_empty_stmt()

    return node_factory.create_func_declaration(declarator, body)
