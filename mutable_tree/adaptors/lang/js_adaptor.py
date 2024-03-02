# https://github.com/tree-sitter/tree-sitter-javascript/blob/master/grammar.js
import tree_sitter
from ...nodes import Expression, Statement
from ...nodes import (
    ArrayAccess,
    ArrayExpression,
    AssignmentExpression,
    BinaryExpression,
    CallExpression,
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
    LambdaExpression,
    SpreadElement,
    AwaitExpression,
    CommaExpression,
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
    ThrowStatement,
    TryStatement,
    FinallyClause,
    TryHandlers,
    WhileStatement,
    YieldStatement,
    WithStatement,
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
from ...nodes import Object, KeyValuePair, ObjectMember, ObjectMembers
from ...nodes import Modifier, ModifierList
from ...nodes import Program
from ...nodes import TypeIdentifier, Dimensions, TypeParameter, TypeParameterList
from ...nodes import (
    get_assignment_op,
    get_binary_op,
    get_unary_op,
    get_update_op,
    get_forin_type,
)
from ...nodes import node_factory
from typing import Tuple, Optional, List


def convert_program(node: tree_sitter.Node) -> Program:
    assert node.type == "program"
    stmts = []
    for ch in node.children:
        stmts.append(convert_statement(ch))

    return node_factory.create_program(node_factory.create_statement_list(stmts))


def convert_statement(node: tree_sitter.Node) -> Statement:
    stmt_convertors = {
        "expression_statement": convert_expression_stmt,
        "statement_block": convert_block_stmt,
        "if_statement": convert_if_stmt,
        "while_statement": convert_while_stmt,
        "do_statement": convert_do_stmt,
        "break_statement": convert_break_stmt,
        "continue_statement": convert_continue_stmt,
        "return_statement": convert_return_stmt,
        "throw_statement": convert_throw_stmt,
        "empty_statement": convert_empty_stmt,
        "labeled_statement": convert_labeled_stmt,
        "switch_statement": convert_switch_stmt,
        "for_statement": convert_for_stmt,
        "for_in_statement": convert_for_in_stmt,
        "try_statement": convert_try_stmt,
        "with_statement": convert_with_stmt,
        "variable_declaration": convert_local_variable_declaration,
        "lexical_declaration": convert_local_variable_declaration,
        "function_declaration": convert_function_declaration,
        "generator_function_declaration": convert_function_declaration,
    }
    return stmt_convertors[node.type](node)


def convert_expression(node: tree_sitter.Node) -> Expression:
    expr_convertors = {
        "identifier": convert_identifier,
        "property_identifier": convert_identifier,
        "statement_identifier": convert_identifier,
        "number": convert_literal,
        "string": convert_literal,
        "template_string": convert_literal,
        "regex": convert_literal,
        "true": convert_literal,
        "false": convert_literal,
        "null": convert_literal,
        "undefined": convert_literal,
        "subscript_expression": convert_subscript_expression,
        "member_expression": convert_member_expression,
        "parenthesized_expression": convert_parenthesized_expression,
        "spread_element": convert_spread_element,
        "array": convert_array,
        "call_expression": convert_call_expression,
        "assignment_expression": convert_assignment_expr,
        "augmented_assignment_expression": convert_augmented_assignment_expr,
        "await_expression": convert_await_expr,
        "unary_expression": convert_unary_expr,
        "binary_expression": convert_binary_expr,
        "ternary_expression": convert_ternary_expr,
        "update_expression": convert_update_expr,
        "new_expression": convert_new_expr,
        "yield_expression": convert_yield_expr,
        "this": convert_this_expr,
        "sequence_expression": convert_sequence_expr,
        "arrow_function": convert_arrow_function,
        "function": convert_function_declaration,
        "object": convert_object,
        # patterns
        "array_pattern": convert_array,
        "object_pattern": convert_object,
    }
    return expr_convertors[node.type](node)


def convert_simple_type(node: tree_sitter.Node) -> TypeIdentifier:
    return node_factory.create_type_identifier(node.text.decode())


def convert_identifier(node: tree_sitter.Node) -> Identifier:
    name = node.text.decode()
    return node_factory.create_identifier(name)


def convert_literal(node: tree_sitter.Node) -> Literal:
    value = node.text.decode()
    return node_factory.create_literal(value)


def convert_subscript_expression(node: tree_sitter.Node) -> ArrayAccess:
    object_expr = convert_expression(node.child_by_field_name("object"))
    optional = node.child_by_field_name("optional_chain") is not None
    index_expr = convert_expression(node.child_by_field_name("index"))

    return node_factory.create_array_access(object_expr, index_expr, optional)


def convert_member_expression(node: tree_sitter.Node) -> FieldAccess:
    object_expr = convert_expression(node.child_by_field_name("object"))
    optional = node.child_by_field_name("optional_chain") is not None
    member_name = convert_expression(node.child_by_field_name("property"))
    return node_factory.create_field_access(object_expr, member_name, optional=optional)


def convert_parenthesized_expression(node: tree_sitter.Node) -> ParenthesizedExpression:
    if node.child_count != 3:
        raise AssertionError("parenthesized_expression should have 3 children")

    expr = convert_expression(node.children[1])
    return node_factory.create_parenthesized_expr(expr)


def convert_spread_element(node: tree_sitter.Node) -> SpreadElement:
    if node.child_count != 2:
        raise AssertionError("spread_element should have 2 children")

    expr = convert_expression(node.children[1])
    return node_factory.create_spread_element(expr)


def convert_array(node: tree_sitter.Node) -> ArrayExpression:
    elements = []
    for ch in node.children[1:-1]:
        if ch.type == ",":
            continue
        elements.append(convert_expression(ch))
    return node_factory.create_array_expr(node_factory.create_expression_list(elements))


def convert_argument_list(node: tree_sitter.Node) -> ExpressionList:
    args = []
    for ch in node.children[1:-1]:
        # skip parenthesis and comma sep
        if ch.type == ",":
            continue
        args.append(convert_expression(ch))
    return node_factory.create_expression_list(args)


def convert_call_expression(node: tree_sitter.Node) -> CallExpression:
    callee_node = node.child_by_field_name("function")
    arg_node = node.child_by_field_name("arguments")
    optional = node.child_by_field_name("optional_chain") is not None

    callee_expr = convert_expression(callee_node)
    args = convert_argument_list(arg_node)

    return node_factory.create_call_expr(callee_expr, args, optional)


def convert_assignment_expr(node: tree_sitter.Node) -> AssignmentExpression:
    if node.child_count != 3:
        raise AssertionError("assignment_expression should have 3 children")

    lhs = convert_expression(node.child_by_field_name("left"))
    rhs = convert_expression(node.child_by_field_name("right"))
    op = get_assignment_op(node.children[1].text.decode())
    return node_factory.create_assignment_expr(lhs, rhs, op)


def convert_augmented_assignment_expr(node: tree_sitter.Node) -> AssignmentExpression:
    lhs = convert_expression(node.child_by_field_name("left"))
    rhs = convert_expression(node.child_by_field_name("right"))
    op = get_assignment_op(node.child_by_field_name("operator").text.decode())
    return node_factory.create_assignment_expr(lhs, rhs, op)


def convert_await_expr(node: tree_sitter.Node) -> AwaitExpression:
    if node.child_count != 2:
        raise AssertionError("await_expression should have 2 children")

    expr = convert_expression(node.children[1])
    return node_factory.create_await_expr(expr)


def convert_unary_expr(node: tree_sitter.Node) -> UnaryExpression:
    op = get_unary_op(node.child_by_field_name("operator").text.decode())
    operand = convert_expression(node.child_by_field_name("argument"))

    if op == "delete":
        return node_factory.create_delete_expr(operand)
    else:
        return node_factory.create_unary_expr(operand, op)


def convert_binary_expr(node: tree_sitter.Node) -> BinaryExpression:
    lhs = convert_expression(node.child_by_field_name("left"))
    rhs = convert_expression(node.child_by_field_name("right"))
    op = get_binary_op(node.child_by_field_name("operator").text.decode())

    return node_factory.create_binary_expr(lhs, rhs, op)


def convert_ternary_expr(node: tree_sitter.Node) -> TernaryExpression:
    condition_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    alternate_node = node.child_by_field_name("alternative")

    condition = convert_expression(condition_node)
    consequence = convert_expression(consequence_node)
    alternate = convert_expression(alternate_node)
    return node_factory.create_ternary_expr(condition, consequence, alternate)


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
    # TODO: type_argument, constructor
    # type_arg_node = node.child_by_field_name('type_arguments')
    type_node = node.child_by_field_name("constructor")
    args_node = node.child_by_field_name("arguments")

    # FIXME: constructor should be primary expression
    type_id = convert_simple_type(type_node)
    args = convert_argument_list(args_node)

    return node_factory.create_new_expr(type_id, args)


def convert_yield_expr(node: tree_sitter.Node) -> YieldStatement:
    if node.child_count == 1:
        return node_factory.create_yield_stmt()
    elif node.child_count == 2:
        expr = convert_expression(node.children[1])
        return node_factory.create_yield_stmt(expr)
    elif node.child_count == 3:
        expr = convert_expression(node.children[2])
        return node_factory.create_yield_stmt(expr, is_delegate=True)
    else:
        raise AssertionError(f"yield_expression has {node.child_count} children")


def convert_this_expr(node: tree_sitter.Node) -> ThisExpression:
    return node_factory.create_this_expr()


def convert_sequence_expr(node: tree_sitter.Node) -> CommaExpression:
    left = convert_expression(node.child_by_field_name("left"))
    right = convert_expression(node.child_by_field_name("right"))

    return node_factory.create_comma_expr(left, right)


def convert_expression_stmt(node: tree_sitter.Node) -> ExpressionStatement:
    expr = convert_expression(node.children[0])
    return node_factory.create_expression_stmt(expr)


def convert_block_stmt(node: tree_sitter.Node) -> BlockStatement:
    stmts = []
    for stmt_node in node.children:
        if stmt_node.type in {"{", "}", ";"}:
            continue
        stmts.append(convert_statement(stmt_node))
    stmts = node_factory.create_statement_list(stmts)
    return node_factory.create_block_stmt(stmts)


def convert_if_stmt(node: tree_sitter.Node) -> IfStatement:
    cond_node = node.child_by_field_name("condition")
    consequence_node = node.child_by_field_name("consequence")
    alternative_node = node.child_by_field_name("alternative")

    assert cond_node.type == "parenthesized_expression" and cond_node.child_count == 3
    cond_node = cond_node.children[1]

    condition = convert_expression(cond_node)
    consequence = convert_statement(consequence_node)
    if alternative_node is None:
        alternative = None
    else:
        assert alternative_node.type == "else_clause"
        # else_clause -> 'else', statement
        alternative = convert_statement(alternative_node.children[1])

    return node_factory.create_if_stmt(condition, consequence, alternative)


def convert_while_stmt(node: tree_sitter.Node) -> WhileStatement:
    cond_node = node.child_by_field_name("condition")
    assert cond_node.child_count == 3, "while condition with != 3 children"
    cond_node = cond_node.children[1]
    body_node = node.child_by_field_name("body")

    cond = convert_expression(cond_node)
    body = convert_statement(body_node)
    return node_factory.create_while_stmt(cond, body)


def convert_do_stmt(node: tree_sitter.Node) -> DoStatement:
    body_node = node.child_by_field_name("body")
    cond_node = node.child_by_field_name("condition")

    assert cond_node.type == "parenthesized_expression"
    assert cond_node.child_count == 3
    cond_node = cond_node.children[1]

    body = convert_statement(body_node)
    cond = convert_expression(cond_node)
    return node_factory.create_do_stmt(cond, body)


def convert_break_stmt(node: tree_sitter.Node) -> BreakStatement:
    label_node = node.child_by_field_name("label")
    label = convert_expression(label_node) if label_node is not None else None
    return node_factory.create_break_stmt(label)


def convert_continue_stmt(node: tree_sitter.Node) -> ContinueStatement:
    label_node = node.child_by_field_name("label")
    label = convert_expression(label_node) if label_node is not None else None
    return node_factory.create_continue_stmt(label)


def convert_return_stmt(node: tree_sitter.Node) -> ReturnStatement:
    # single return
    if node.child_count == 1:
        return node_factory.create_return_stmt()

    # return;
    if node.children[1].type == ";":
        return node_factory.create_return_stmt()

    # return expr(;)
    expr_node = node.children[1]
    expr = convert_expression(expr_node) if expr_node is not None else None
    return node_factory.create_return_stmt(expr)


def convert_throw_stmt(node: tree_sitter.Node) -> ThrowStatement:
    assert node.child_count in {2, 3}
    expr_node = node.children[1]
    expr = convert_expression(expr_node)
    return node_factory.create_throw_stmt(expr)


def convert_empty_stmt(node: tree_sitter.Node) -> EmptyStatement:
    return node_factory.create_empty_stmt()


def convert_labeled_stmt(node: tree_sitter.Node) -> LabeledStatement:
    label_node = node.child_by_field_name("label")
    body_node = node.child_by_field_name("body")

    label = convert_expression(label_node)
    body = convert_statement(body_node)
    return node_factory.create_labeled_stmt(label, body)


def convert_switch_case(node: tree_sitter.Node) -> SwitchCase:
    value_node = node.child_by_field_name("value")
    body_nodes = node.children_by_field_name("body")  # repeat(statement)

    if value_node is None:
        # switch_default
        label = None
    else:
        label = convert_expression(value_node)

    stmts = [convert_statement(s) for s in body_nodes]
    stmt_list = node_factory.create_statement_list(stmts)
    return node_factory.create_switch_case(stmt_list, label)


def convert_switch_block(node: tree_sitter.Node) -> SwitchCaseList:
    # ignore curly braces
    cases = []
    for child in node.children[1:-1]:
        cases.append(convert_switch_case(child))

    return node_factory.create_switch_case_list(cases)


def convert_switch_stmt(node: tree_sitter.Node) -> SwitchStatement:
    cond_node = node.child_by_field_name("value")
    body_node = node.child_by_field_name("body")

    assert cond_node.type == "parenthesized_expression" and cond_node.child_count == 3
    cond_node = cond_node.children[1]

    cond = convert_expression(cond_node)
    body = convert_switch_block(body_node)

    return node_factory.create_switch_stmt(cond, body)


def convert_variable_declarator(node: tree_sitter.Node) -> Declarator:
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")

    if name_node.type == "identifier":
        name = convert_identifier(name_node)
        decl = node_factory.create_variable_declarator(name)
    else:
        pattern = convert_expression(name_node)
        decl = node_factory.create_destructuring_declarator(pattern)

    value = convert_expression(value_node) if value_node is not None else None
    if value is not None:
        decl = node_factory.create_initializing_declarator(decl, value)

    return decl


def convert_local_variable_declaration(
    node: tree_sitter.Node,
) -> LocalVariableDeclaration:
    decl_kind = node.child_by_field_name("kind")
    if decl_kind is None:
        assert node.children[0].type == "var"
        decl_kind = "var"
    else:
        decl_kind = decl_kind.text.decode()
    # NOTE: as a workaround we are treating 'let', 'var' etc as type identifiers
    decl_kind_ty = node_factory.create_type_identifier(decl_kind)
    decl_type = node_factory.create_declarator_type(decl_kind_ty)

    declarators = []
    for decl_node in node.children[1:]:
        if decl_node.type == ";" or decl_node.type == ",":
            continue
        declarators.append(convert_variable_declarator(decl_node))

    declarators = node_factory.create_declarator_list(declarators)

    return node_factory.create_local_variable_declaration(decl_type, declarators)


def convert_sequence_expr_to_list(node: tree_sitter.Node) -> List[Expression]:
    assert node.type == "sequence_expression"
    exprs = []

    def _convert_sequence_expr_to_list(node: tree_sitter.Node):
        if node.type == "sequence_expression":
            exprs.append(convert_expression(node.child_by_field_name("left")))
            _convert_sequence_expr_to_list(node.child_by_field_name("right"))
        else:
            exprs.append(convert_expression(node))

    _convert_sequence_expr_to_list(node)
    return exprs


def convert_for_stmt(node: tree_sitter.Node) -> ForStatement:
    init_node = node.child_by_field_name("initializer")
    cond_node = node.child_by_field_name("condition")
    update_node = node.child_by_field_name("increment")
    body_node = node.child_by_field_name("body")

    body = convert_statement(body_node)

    if init_node.type == "empty_statement":
        init = None
    else:
        if init_node.type in {"lexical_declaration", "variable_declaration"}:
            init = convert_local_variable_declaration(init_node)
        else:
            # single or comma expression
            assert init_node.type == "expression_statement"
            init_node = init_node.children[0]

            if init_node.type == "sequence_expression":
                init_exprs = convert_sequence_expr_to_list(init_node)
                init = node_factory.create_expression_list(init_exprs)
            else:
                init = convert_expression(init_node)
                init = node_factory.create_expression_list([init])

    if cond_node.type == "empty_statement":
        cond = None
    else:
        assert cond_node.type == "expression_statement", cond_node.type
        cond = convert_expression(cond_node.children[0])

    if update_node is None:
        update = None
    else:
        if update_node.type == "sequence_expression":
            update = convert_sequence_expr_to_list(update_node)
            update = node_factory.create_expression_list(update)
        else:
            update = convert_expression(update_node)
            update = node_factory.create_expression_list([update])

    return node_factory.create_for_stmt(body, init, cond, update)


def convert_for_in_stmt(node: tree_sitter.Node) -> ForInStatement:
    is_async = node.children[1].type == "await"

    kind_node = node.child_by_field_name("kind")
    left_node = node.child_by_field_name("left")
    operator = node.child_by_field_name("operator").text.decode()
    right_node = node.child_by_field_name("right")
    body_node = node.child_by_field_name("body")

    if kind_node is None:
        # FIXME: temporary workaround
        kind_id = node_factory.create_type_identifier("let")
    else:
        kind_id = node_factory.create_type_identifier(kind_node.text.decode())
    decl_ty = node_factory.create_declarator_type(kind_id)

    if left_node.type != "identifier":
        left = convert_expression(left_node)
    else:
        left = convert_identifier(left_node)
        left = node_factory.create_variable_declarator(left)
    forin_type = get_forin_type(operator)
    right = convert_expression(right_node)
    body = convert_statement(body_node)

    return node_factory.create_for_in_stmt(
        decl_ty, left, right, body, forin_type, is_async
    )


def convert_try_handler(node: tree_sitter.Node) -> TryHandlers:
    parameter_node = node.child_by_field_name("parameter")
    body_node = node.child_by_field_name("body")

    if parameter_node is None:
        parameter = []
    else:
        if parameter_node.type != "identifier":
            raise NotImplementedError("try_handler with non-identifier parameter")
        else:
            parameter = convert_identifier(parameter_node)

    body = convert_statement(body_node)

    catch_clause = node_factory.create_catch_clause(body, exception=parameter)
    return node_factory.create_try_handlers([catch_clause])


def convert_try_finalizer(node: tree_sitter.Node) -> FinallyClause:
    body_node = node.child_by_field_name("body")
    body = convert_statement(body_node)
    return node_factory.create_finally_clause(body)


def convert_try_stmt(node: tree_sitter.Node) -> TryStatement:
    body_node = node.child_by_field_name("body")
    handler_node = node.child_by_field_name("handler")
    finalizer_node = node.child_by_field_name("finalizer")

    body = convert_statement(body_node)
    handler = convert_try_handler(handler_node) if handler_node is not None else None
    finalizer = (
        convert_try_finalizer(finalizer_node) if finalizer_node is not None else None
    )

    return node_factory.create_try_stmt(body, handler, finalizer)


def convert_with_stmt(node: tree_sitter.Node) -> WithStatement:
    object_node = node.child_by_field_name("object")
    body_node = node.child_by_field_name("body")

    assert object_node.type == "parenthesized_expression"
    object = convert_expression(object_node.children[1])
    body = convert_statement(body_node)

    return node_factory.create_with_stmt(object, body)


def _convert_formal_param(node: tree_sitter.Node) -> UntypedParameter:
    assert node.type == "identifier"
    decl_id = convert_identifier(node)
    param_decl = node_factory.create_variable_declarator(decl_id)
    return node_factory.create_untyped_param(param_decl)


def _convert_default_formal_param(node: tree_sitter.Node) -> UntypedParameter:
    left_node = node.child_by_field_name("left")
    right_node = node.child_by_field_name("right")

    assert left_node is not None and left_node.type == "identifier"
    left_id = convert_identifier(left_node)
    left = node_factory.create_variable_declarator(left_id)
    right = convert_expression(right_node)

    param_decl = node_factory.create_initializing_declarator(left, right)
    return node_factory.create_untyped_param(param_decl)


def convert_formal_param(node: tree_sitter.Node) -> FormalParameter:
    if node.child_count == 0:
        return _convert_formal_param(node)
    elif node.child_count == 3:
        return _convert_default_formal_param(node)
    else:
        raise AssertionError(f"formal_param with {node.child_count} children")


def convert_formal_parameters(node: tree_sitter.Node) -> FormalParameterList:
    params = []
    for child in node.children[1:-1]:
        if child.type == ",":
            continue
        else:
            params.append(convert_formal_param(child))
    return node_factory.create_formal_parameter_list(params)


def convert_function_header(node: tree_sitter.Node) -> FunctionHeader:
    # modifiers
    modifiers = []

    # decorators
    decorator_nodes = node.children_by_field_name("decorator")
    for decorator_node in decorator_nodes:
        modifiers.append(node_factory.create_modifier(decorator_node.text.decode()))

    # static, async, get, set
    idx = 0
    while node.children[idx].type in {"static", "async", "get", "set"}:
        modifiers.append(node_factory.create_modifier(node.children[idx].text.decode()))
        idx += 1

    if len(modifiers) == 0:
        modifiers = None
    else:
        modifiers = node_factory.create_modifier_list(modifiers)

    if node.children[idx].type == "function":
        idx += 1

    if node.children[idx].type == "*":
        raise NotImplementedError("generator function")

    # function name
    name_node = node.child_by_field_name("name")
    if name_node is None:
        name = node_factory.create_anonymous_declarator()
    else:
        assert name_node.type == "identifier"
        name_identifier = convert_identifier(name_node)
        name = node_factory.create_variable_declarator(name_identifier)

    # parameters
    params_node = node.child_by_field_name("parameters")
    params = convert_formal_parameters(params_node)
    func_decl = node_factory.create_func_declarator(name, params)

    return node_factory.create_func_header(func_decl, modifiers=modifiers)


def convert_function_declaration(node: tree_sitter.Node) -> FunctionDeclaration:
    header = convert_function_header(node)
    body = convert_statement(node.child_by_field_name("body"))

    return node_factory.create_func_declaration(header, body)


def convert_arrow_function(node: tree_sitter.Node) -> LambdaExpression:
    if node.children[0].type == "async":
        modifiers = node_factory.create_modifier_list(
            [node_factory.create_modifier("async")]
        )
    else:
        modifiers = None

    param_node = node.child_by_field_name("parameter")
    if param_node is not None:
        parenthesized = False
        params = []
        params.append(convert_formal_param(param_node))
        params = node_factory.create_formal_parameter_list(params)
    else:
        param_node = node.child_by_field_name("parameters")
        assert param_node is not None
        parenthesized = True
        params = convert_formal_parameters(param_node)

    body_node = node.child_by_field_name("body")
    if body_node.type == "statement_block":
        body = convert_statement(body_node)
    else:
        body = convert_expression(body_node)

    return node_factory.create_lambda_expr(params, body, parenthesized, modifiers)


def convert_key_value_pair(node: tree_sitter.Node) -> KeyValuePair:
    key_node = node.child_by_field_name("key")
    value_node = node.child_by_field_name("value")

    if key_node.type == "computed_property_name":
        key = convert_expression(key_node.children[1])
        key = node_factory.create_computed_property_name(key)
    else:
        key = convert_expression(key_node)

    value = convert_expression(value_node)
    return node_factory.create_key_value_pair(key, value)


def convert_object_member(node: tree_sitter.Node) -> ObjectMember:
    convertors = {
        "pair": convert_key_value_pair,
        "spread_element": convert_spread_element,
        "method_definition": convert_function_declaration,
        "shorthand_property_identifier": convert_identifier,
        "shorthand_property_identifier_pattern": convert_identifier,
    }
    return convertors[node.type](node)


def convert_object(node: tree_sitter.Node) -> Object:
    members = []
    for child in node.children[1:-1]:
        if child.type == ",":
            continue
        members.append(convert_object_member(child))

    members = node_factory.create_object_members(members)
    return node_factory.create_object(members)
