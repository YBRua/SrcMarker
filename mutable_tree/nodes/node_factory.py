from .node import NodeType
from .program import Program
from .expressions import Expression
from .expressions import (
    BinaryOps,
    UnaryOps,
    UpdateOps,
    AssignmentOps,
    FieldAccessOps,
    PointerOps,
)
from .expressions import (
    ArrayAccess,
    ArrayExpression,
    ArrayCreationExpression,
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
    PrimaryExpression,
    ParenthesizedExpression,
    ExpressionList,
    CommaExpression,
    SizeofExpression,
    PointerExpression,
    DeleteExpression,
    ScopeResolution,
    QualifiedIdentifier,
    CompoundLiteralExpression,
    SpreadElement,
    AwaitExpression,
)
from .statements import Statement
from .statements import (
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
    SwitchCase,
    SwitchCaseList,
    SwitchStatement,
    ThrowStatement,
    TryStatement,
    TryHandlers,
    CatchClause,
    FinallyClause,
    WhileStatement,
    YieldStatement,
    StatementList,
    TryResource,
    TryResourceList,
    TryWithResourcesStatement,
    SynchronizedStatement,
    LambdaExpression,
    GotoStatement,
    WithStatement,
)
from .statements import ForInType
from .statements import (
    Declarator,
    VariableDeclarator,
    ArrayDeclarator,
    PointerDeclarator,
    ReferenceDeclarator,
    InitializingDeclarator,
    AnonymousDeclarator,
    DeclaratorList,
    DeclaratorType,
    LocalVariableDeclaration,
    DestructuringDeclarator,
)
from .statements import (
    FormalParameter,
    UntypedParameter,
    TypedFormalParameter,
    SpreadParameter,
    VariadicParameter,
    FormalParameterList,
    FunctionDeclarator,
    FunctionHeader,
    FunctionDeclaration,
)
from .statements import (
    TemplateDeclaration,
    TemplateParameter,
    TemplateParameterList,
    TypeParameterDeclaration,
    TypenameOpts,
)
from .classes import (
    KeyValuePair,
    ObjectMembers,
    ObjectMember,
    Object,
    ComputedPropertyName,
)
from .miscs import Modifier, ModifierList
from .statements.for_stmt import ForInit
from .types import (
    TypeIdentifier,
    TypeIdentifierList,
    DimensionSpecifier,
    Dimensions,
    TypeParameter,
    TypeParameterList,
)

from typing import Union, Optional, List

# TOP LEVEL


def create_program(stmts: StatementList) -> Program:
    return Program(NodeType.PROGRAM, stmts)


# TYPES


def create_array_type(
    type_identifier: TypeIdentifier, dimension: Dimensions
) -> TypeIdentifier:
    type_identifier.dimension = dimension
    return type_identifier


def create_type_identifier(
    name: str,
    dimension: Optional[Dimensions] = None,
) -> TypeIdentifier:
    return TypeIdentifier(NodeType.TYPE_IDENTIFIER, name, dimension)


def create_dimension_specifier(expr: Optional[Expression] = None) -> DimensionSpecifier:
    return DimensionSpecifier(NodeType.DIMENSION_SPECIFIER, expr)


def create_dimensions(dims: List[DimensionSpecifier]) -> Dimensions:
    return Dimensions(NodeType.DIMENSIONS, dims)


def create_type_parameter(
    name: str,
    extends: Optional[TypeIdentifierList] = None,
) -> TypeParameter:
    return TypeParameter(NodeType.TYPE_PARAMETER, name, extends)


def create_type_parameter_list(type_params: List[TypeParameter]) -> TypeParameterList:
    return TypeParameterList(NodeType.TYPE_PARAMETER_LIST, type_params)


# EXPRESSIONS


def create_identifier(name: str) -> Identifier:
    return Identifier(NodeType.IDENTIFIER, name)


def create_literal(value: str) -> Literal:
    return Literal(NodeType.LITERAL, value)


def create_assignment_expr(
    left: Union[Identifier, ArrayAccess, FieldAccess],
    right: Expression,
    op: AssignmentOps,
) -> AssignmentExpression:
    return AssignmentExpression(NodeType.ASSIGNMENT_EXPR, left, right, op)


def create_binary_expr(
    left: Expression, right: Expression, op: BinaryOps
) -> BinaryExpression:
    return BinaryExpression(NodeType.BINARY_EXPR, left, right, op)


def create_unary_expr(expr: Expression, op: UnaryOps) -> UnaryExpression:
    return UnaryExpression(NodeType.UNARY_EXPR, expr, op)


def create_update_expr(
    expr: Union[Identifier, FieldAccess], op: UpdateOps, prefix: bool
) -> UpdateExpression:
    return UpdateExpression(NodeType.UPDATE_EXPR, expr, op, prefix)


def create_array_access(
    array: PrimaryExpression, index: Expression, optional: bool = False
) -> ArrayAccess:
    return ArrayAccess(NodeType.ARRAY_ACCESS, array, index, optional)


def create_array_expr(elements: ExpressionList) -> ArrayExpression:
    return ArrayExpression(NodeType.ARRAY_EXPR, elements)


def create_call_expr(
    callee: PrimaryExpression, args: ExpressionList, optional: bool = False
) -> CallExpression:
    return CallExpression(NodeType.CALL_EXPR, callee, args, optional)


def create_cast_expr(type_name: TypeIdentifier, expr: Expression) -> CastExpression:
    return CastExpression(NodeType.CAST_EXPR, type_name, expr)


def create_field_access(
    obj: PrimaryExpression,
    field: Identifier,
    op: FieldAccessOps = FieldAccessOps.DOT,
    optional: bool = False,
) -> FieldAccess:
    return FieldAccess(NodeType.FIELD_ACCESS, obj, field, op, optional)


def create_instanceof_expr(
    expr: Expression, type_name: TypeIdentifier
) -> InstanceofExpression:
    return InstanceofExpression(NodeType.INSTANCEOF_EXPR, expr, type_name)


def create_new_expr(
    type_name: TypeIdentifier, args: Optional[ExpressionList] = None
) -> NewExpression:
    return NewExpression(NodeType.NEW_EXPR, type_name, args)


def create_array_creation_expr(
    type_name: TypeIdentifier,
    dimensions: Dimensions,
    initializer: Optional[ArrayExpression] = None,
) -> ArrayCreationExpression:
    return ArrayCreationExpression(
        NodeType.ARRAY_CREATION_EXPR, type_name, dimensions, initializer
    )


def create_ternary_expr(
    condition: Expression, consequence: Expression, alternate: Expression
) -> TernaryExpression:
    return TernaryExpression(NodeType.TERNARY_EXPR, condition, consequence, alternate)


def create_this_expr() -> ThisExpression:
    return ThisExpression(NodeType.THIS_EXPR)


def create_parenthesized_expr(expr: Expression) -> Expression:
    return ParenthesizedExpression(NodeType.PARENTHESIZED_EXPR, expr)


def create_lambda_expr(
    params: FormalParameterList,
    body: Union[Expression, BlockStatement],
    parenthesized: bool = False,
    modifiers: Optional[ModifierList] = None,
) -> LambdaExpression:
    return LambdaExpression(
        NodeType.LAMBDA_EXPR, params, body, parenthesized, modifiers
    )


def create_comma_expr(left: Expression, right: Expression) -> CommaExpression:
    return CommaExpression(NodeType.COMMA_EXPR, left, right)


def create_sizeof_expr(operand: Union[Expression, TypeIdentifier]) -> SizeofExpression:
    return SizeofExpression(NodeType.SIZEOF_EXPR, operand)


def create_pointer_expr(operand: Expression, op: PointerOps) -> PointerExpression:
    return PointerExpression(NodeType.POINTER_EXPR, operand, op)


def create_delete_expr(operand: Expression, is_array: bool = False) -> DeleteExpression:
    return DeleteExpression(NodeType.DELETE_EXPR, operand, is_array)


def create_scope_resolution(
    scope: Optional[Union[Identifier, TypeIdentifier]] = None
) -> ScopeResolution:
    return ScopeResolution(NodeType.SCOPE_RESOLUTION, scope)


def create_qualified_identifier(
    scope: ScopeResolution,
    name: Union[Identifier, QualifiedIdentifier, TypeIdentifier],
) -> QualifiedIdentifier:
    return QualifiedIdentifier(NodeType.QUALIFIED_IDENTIFIER, scope, name)


def create_compound_literal_expr(
    type_id: TypeIdentifier,
    value: ArrayExpression,
) -> CompoundLiteralExpression:
    return CompoundLiteralExpression(NodeType.COMPOUND_LITERAL_EXPR, type_id, value)


# DECLARATIONS


def create_variable_declarator(decl_id: Identifier) -> VariableDeclarator:
    return VariableDeclarator(NodeType.VARIABLE_DECLARATOR, decl_id)


def create_array_declarator(
    decl: Declarator, dim: DimensionSpecifier
) -> ArrayDeclarator:
    return ArrayDeclarator(NodeType.ARRAY_DECLARATOR, decl, dim)


def create_pointer_declarator(decl: Declarator) -> PointerDeclarator:
    return PointerDeclarator(NodeType.POINTER_DECLARATOR, decl)


def create_reference_declarator(
    decl: Declarator, r_ref: bool = False
) -> ReferenceDeclarator:
    return ReferenceDeclarator(NodeType.REFERENCE_DECLARATOR, decl, r_ref)


def create_initializing_declarator(
    decl: Declarator, value: Expression
) -> InitializingDeclarator:
    return InitializingDeclarator(NodeType.INITIALIZING_DECLARATOR, decl, value)


def create_declarator_type(
    type_id: TypeIdentifier,
    prefixes: Optional[ModifierList] = None,
    postfixes: Optional[ModifierList] = None,
) -> DeclaratorType:
    return DeclaratorType(NodeType.DECLARATOR_TYPE, type_id, prefixes, postfixes)


def create_local_variable_declaration(
    decl_type: DeclaratorType, declarators: DeclaratorList
) -> LocalVariableDeclaration:
    return LocalVariableDeclaration(
        NodeType.LOCAL_VARIABLE_DECLARATION, decl_type, declarators
    )


def create_anonymous_declarator() -> AnonymousDeclarator:
    return AnonymousDeclarator(NodeType.ANONYMOUS_DECLARATOR)


def create_destructuring_declarator(pattern: Expression) -> DestructuringDeclarator:
    return DestructuringDeclarator(NodeType.DESTRUCTURING_DECLARATOR, pattern)


# STATEMENTS


def create_empty_stmt() -> EmptyStatement:
    return EmptyStatement(NodeType.EMPTY_STMT)


def create_expression_stmt(expr: Expression) -> ExpressionStatement:
    return ExpressionStatement(NodeType.EXPRESSION_STMT, expr)


def create_for_stmt(
    body: Statement,
    init: Optional[ForInit] = None,
    condition: Optional[Expression] = None,
    update: Optional[ExpressionList] = None,
) -> ForStatement:
    return ForStatement(NodeType.FOR_STMT, body, init, condition, update)


def create_while_stmt(condition: Expression, body: Statement) -> WhileStatement:
    return WhileStatement(NodeType.WHILE_STMT, condition, body)


def create_block_stmt(statements: StatementList) -> BlockStatement:
    return BlockStatement(NodeType.BLOCK_STMT, statements)


def create_assert_stmt(
    condition: Expression, message: Optional[Expression] = None
) -> AssertStatement:
    return AssertStatement(NodeType.ASSERT_STMT, condition, message)


def create_break_stmt(label: Optional[Identifier] = None) -> BreakStatement:
    return BreakStatement(NodeType.BREAK_STMT, label)


def create_continue_stmt(label: Optional[Identifier] = None) -> ContinueStatement:
    return ContinueStatement(NodeType.CONTINUE_STMT, label)


def create_do_stmt(condition: Expression, body: Statement) -> DoStatement:
    return DoStatement(NodeType.DO_STMT, body, condition)


def create_for_in_stmt(
    decl_type: DeclaratorType,
    decl: Union[Declarator, Expression],
    iterable: Expression,
    body: Statement,
    forin_type: ForInType = ForInType.COLON,
    is_async: bool = False,
) -> ForInStatement:
    return ForInStatement(
        NodeType.FOR_IN_STMT, decl_type, decl, iterable, body, forin_type, is_async
    )


def create_if_stmt(
    condition: Expression,
    consequence: Statement,
    alternate: Optional[Statement] = None,
) -> IfStatement:
    return IfStatement(NodeType.IF_STMT, condition, consequence, alternate)


def create_labeled_stmt(label: Identifier, stmt: Statement) -> LabeledStatement:
    return LabeledStatement(NodeType.LABELED_STMT, label, stmt)


def create_return_stmt(value: Optional[Expression] = None) -> ReturnStatement:
    return ReturnStatement(NodeType.RETURN_STMT, value)


def create_switch_case(
    stmts: StatementList,
    case: Optional[Expression] = None,
) -> SwitchCase:
    return SwitchCase(NodeType.SWITCH_CASE, stmts, case)


def create_switch_case_list(cases: List[SwitchCase]) -> SwitchCaseList:
    return SwitchCaseList(NodeType.SWITCH_CASE_LIST, cases)


def create_switch_stmt(
    condition: Expression,
    cases: SwitchCaseList,
) -> SwitchStatement:
    return SwitchStatement(NodeType.SWITCH_STMT, condition, cases)


def create_throw_stmt(expr: Expression) -> ThrowStatement:
    return ThrowStatement(NodeType.THROW_STMT, expr)


def create_yield_stmt(
    expr: Optional[Expression] = None, is_delegate: bool = False
) -> YieldStatement:
    return YieldStatement(NodeType.YIELD_STMT, expr, is_delegate)


def create_catch_clause(
    body: BlockStatement,
    exception_types: Optional[TypeIdentifierList] = None,
    exception: Optional[Identifier] = None,
    modifiers: Optional[ModifierList] = None,
) -> CatchClause:
    return CatchClause(
        NodeType.CATCH_CLAUSE, body, exception_types, exception, modifiers
    )


def create_finally_clause(body: BlockStatement) -> FinallyClause:
    return FinallyClause(NodeType.FINALLY_CLAUSE, body)


def create_try_handlers(catch_clauses: List[CatchClause]) -> TryHandlers:
    return TryHandlers(NodeType.TRY_HANDLERS, catch_clauses)


def create_try_stmt(
    try_block: BlockStatement,
    handlers: Optional[TryHandlers] = None,
    finally_clause: Optional[FinallyClause] = None,
) -> TryStatement:
    return TryStatement(NodeType.TRY_STMT, try_block, handlers, finally_clause)


def create_try_resource(
    resource: Union[Identifier, FieldAccess, LocalVariableDeclaration]
) -> TryResource:
    return TryResource(NodeType.TRY_RESOURCE, resource)


def create_try_resource_list(resources: List[TryResource]) -> TryResourceList:
    return TryResourceList(NodeType.TRY_RESOURCE_LIST, resources)


def create_try_with_resources_stmt(
    resources: TryResourceList,
    try_block: BlockStatement,
    handlers: TryHandlers,
    finally_clause: Optional[FinallyClause] = None,
) -> TryWithResourcesStatement:
    return TryWithResourcesStatement(
        NodeType.TRY_WITH_RESOURCES_STMT, resources, try_block, handlers, finally_clause
    )


def create_synchronized_stmt(
    expr: ParenthesizedExpression, body: BlockStatement
) -> SynchronizedStatement:
    return SynchronizedStatement(NodeType.SYNCHRONIZED_STMT, expr, body)


def wrap_block_stmt(stmt: Statement) -> BlockStatement:
    return BlockStatement(
        NodeType.BLOCK_STMT, StatementList(NodeType.STATEMENT_LIST, [stmt])
    )


def create_goto_stmt(label: Identifier) -> GotoStatement:
    return GotoStatement(NodeType.GOTO_STMT, label)


def create_with_stmt(object: Expression, body: Statement) -> WithStatement:
    return WithStatement(NodeType.WITH_STMT, object, body)


# DECLARATIONS & DEFINITIONS


def create_untyped_param(decl: Declarator) -> UntypedParameter:
    return UntypedParameter(NodeType.UNTYPED_PARAMETER, decl)


def create_typed_formal_param(
    decl_type: DeclaratorType, decl: Optional[Declarator] = None
) -> TypedFormalParameter:
    return TypedFormalParameter(NodeType.FORMAL_PARAMETER, decl_type, decl)


def create_spread_param(decl: Declarator, decl_type: DeclaratorType) -> SpreadParameter:
    return SpreadParameter(NodeType.SPREAD_PARAMETER, decl, decl_type)


def create_formal_parameter_list(params: List[FormalParameter]) -> FormalParameterList:
    return FormalParameterList(NodeType.FORMAL_PARAMETER_LIST, params)


def create_variadic_parameter() -> VariadicParameter:
    return VariadicParameter(NodeType.VARIADIC_PARAMETER)


def create_func_declarator(
    decl: Declarator, params: FormalParameterList
) -> FunctionDeclarator:
    return FunctionDeclarator(NodeType.FUNCTION_DECLARATOR, decl, params)


def create_func_header(
    func_decl: FunctionDeclarator,
    return_type: Optional[DeclaratorType] = None,
    dimensions: Optional[Dimensions] = None,
    throws: Optional[TypeIdentifierList] = None,
    modifiers: Optional[ModifierList] = None,
    type_params: Optional[TypeParameterList] = None,
) -> FunctionHeader:
    return FunctionHeader(
        NodeType.FUNCTION_HEADER,
        func_decl,
        return_type,
        dimensions,
        throws,
        modifiers,
        type_params,
    )


def create_func_declaration(
    header: FunctionHeader,
    body: Union[BlockStatement, EmptyStatement],
) -> FunctionDeclaration:
    return FunctionDeclaration(NodeType.FUNCTION_DEFINITION, header, body)


def create_type_parameter_declaration(
    type_id: TypeIdentifier,
    typename_opt: TypenameOpts,
) -> TypeParameterDeclaration:
    return TypeParameterDeclaration(
        NodeType.TYPE_PARAMETER_DECLARATION, type_id, typename_opt
    )


def create_template_parameter_list(
    params: List[TemplateParameter],
) -> TemplateParameterList:
    return TemplateParameterList(NodeType.TEMPLATE_PARAMETER_LIST, params)


def create_template_declaration(
    params: TemplateParameterList, func: FunctionDeclaration
) -> TemplateDeclaration:
    return TemplateDeclaration(NodeType.TEMPLATE_DECLARATION, params, func)


# OBJECTS & CLASSES


def create_computed_property_name(expr: Expression) -> ComputedPropertyName:
    return ComputedPropertyName(NodeType.COMPUTED_PROPERTY_NAME, expr)


def create_key_value_pair(key: Identifier, value: Expression) -> KeyValuePair:
    return KeyValuePair(NodeType.KEYVALUE_PAIR, key, value)


def create_object_members(members: List[ObjectMember]) -> ObjectMembers:
    return ObjectMembers(NodeType.OBJECT_MEMBERS, members)


def create_object(members: ObjectMembers) -> Object:
    return Object(NodeType.OBJECT, members)


# MISCS


def create_type_identifier_list(type_ids: List[TypeIdentifier]) -> TypeIdentifierList:
    return TypeIdentifierList(NodeType.TYPE_IDENTIFIER_LIST, type_ids)


def create_expression_list(exprs: List[Expression]) -> ExpressionList:
    return ExpressionList(NodeType.EXPRESSION_LIST, exprs)


def create_statement_list(stmts: List[Statement]) -> StatementList:
    return StatementList(NodeType.STATEMENT_LIST, stmts)


def create_declarator_list(declarators: List[Declarator]) -> DeclaratorList:
    return DeclaratorList(NodeType.DECLARATOR_LIST, declarators)


def create_modifier(name: str) -> Modifier:
    return Modifier(NodeType.MODIFIER, name)


def create_modifier_list(modifiers: List[Modifier]) -> ModifierList:
    return ModifierList(NodeType.MODIFIER_LIST, modifiers)


def create_spread_element(expr: Expression) -> SpreadElement:
    return SpreadElement(NodeType.SPREAD_ELEMENT, expr)


def create_await_expr(expr: Expression) -> AwaitExpression:
    return AwaitExpression(NodeType.AWAIT_EXPR, expr)
