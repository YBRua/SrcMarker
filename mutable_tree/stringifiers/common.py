from ..nodes import Node, is_expression
from ..nodes import (
    ArrayAccess,
    ArrayCreationExpression,
    ArrayExpression,
    AssignmentExpression,
    BinaryExpression,
    CallExpression,
    CastExpression,
    CommaExpression,
    CompoundLiteralExpression,
    DeleteExpression,
    FieldAccess,
    Identifier,
    InstanceofExpression,
    Literal,
    NewExpression,
    ParenthesizedExpression,
    PointerExpression,
    QualifiedIdentifier,
    SizeofExpression,
    TernaryExpression,
    ThisExpression,
    UnaryExpression,
    UpdateExpression,
)
from ..nodes import (
    AssertStatement,
    BlockStatement,
    BreakStatement,
    ContinueStatement,
    DoStatement,
    EmptyStatement,
    ExpressionStatement,
    ForInStatement,
    ForStatement,
    GotoStatement,
    IfStatement,
    LabeledStatement,
    LambdaExpression,
    ReturnStatement,
    SwitchStatement,
    SynchronizedStatement,
    ThrowStatement,
    TryStatement,
    TryWithResourcesStatement,
    WhileStatement,
    YieldStatement,
)
from ..nodes import (
    DimensionSpecifier,
    Dimensions,
    TypeIdentifier,
    TypeParameterList,
    TypeParameter,
)
from ..nodes import (
    InitializingDeclarator,
    VariableDeclarator,
    PointerDeclarator,
    ReferenceDeclarator,
    ArrayDeclarator,
    VariadicParameter,
    UntypedParameter,
    TypedFormalParameter,
    SpreadParameter,
    FunctionDeclarator,
    FunctionHeader,
    FunctionDeclaration,
    DeclaratorType,
    LocalVariableDeclaration,
    TypeParameterDeclaration,
    TemplateParameterList,
    TemplateDeclaration,
)
from ..nodes import Modifier, ModifierList
from ..nodes import ScopeResolution, SwitchCase, TryResource, CatchClause, FinallyClause
from ..nodes import Program

from ..nodes import (
    SpreadElement,
    AwaitExpression,
    WithStatement,
    AnonymousDeclarator,
    KeyValuePair,
    Object,
    DestructuringDeclarator,
    ComputedPropertyName,
)


class BaseStringifier:
    def stringify(self, node: Node) -> str:
        stringifier = getattr(self, f"stringify_{type(node).__name__}", None)
        if stringifier is None:
            raise NotImplementedError(f"No stringifier for {type(node).__name__}")
        return stringifier(node)

    def stringify_Program(self, node: Program) -> str:
        return "\n".join([self.stringify(s) for s in node.main.get_children()])

    def stringify_ArrayAccess(self, node: ArrayAccess) -> str:
        return f"{self.stringify(node.array)}[{self.stringify(node.index)}]"

    def stringify_ArrayCreationExpression(self, node: ArrayCreationExpression) -> str:
        type_str = self.stringify(node.type_id)
        dims_str = self.stringify(node.dimensions)
        if node.value is not None:
            value_str = self.stringify(node.value)
            return f"new {type_str}{dims_str} {value_str}"
        else:
            return f"new {type_str}{dims_str}"

    def stringify_ArrayExpression(self, node: ArrayExpression) -> str:
        children_strs = [self.stringify(elem) for elem in node.elements.get_children()]
        return f'{{{", ".join(children_strs)}}}'

    def stringify_AssignmentExpression(self, node: AssignmentExpression) -> str:
        return (
            f"{self.stringify(node.left)} {node.op.value} {self.stringify(node.right)}"
        )

    def stringify_BinaryExpression(self, node: BinaryExpression) -> str:
        return (
            f"{self.stringify(node.left)} {node.op.value} {self.stringify(node.right)}"
        )

    def stringify_CallExpression(self, node: CallExpression) -> str:
        arg_list_strs = [self.stringify(arg) for arg in node.args.get_children()]
        arg_list = ", ".join(arg_list_strs)
        return f"{self.stringify(node.callee)}({arg_list})"

    def stringify_CastExpression(self, node: CastExpression) -> str:
        return f"({self.stringify(node.type)}) {self.stringify(node.value)}"

    def stringify_CommaExpression(self, node: CommaExpression) -> str:
        return f"{self.stringify(node.left)}, {self.stringify(node.right)}"

    def stringify_CompoundLiteralExpression(
        self, node: CompoundLiteralExpression
    ) -> str:
        return f"({self.stringify(node.type_id)}) {self.stringify(node.value)}"

    def stringify_DeleteExpression(self, node: DeleteExpression) -> str:
        if node.is_array:
            return f"delete[] {self.stringify(node.operand)}"
        else:
            return f"delete {self.stringify(node.operand)}"

    def stringify_FieldAccess(self, node: FieldAccess) -> str:
        return (
            f"{self.stringify(node.object)}{node.op.value}{self.stringify(node.field)}"
        )

    def stringify_Identifier(self, node: Identifier) -> str:
        return str(node.name)

    def stringify_InstanceofExpression(self, node: InstanceofExpression) -> str:
        return f"{self.stringify(node.left)} instanceof {self.stringify(node.right)}"

    def stringify_Literal(self, node: Literal) -> str:
        return str(node.value)

    def stringify_NewExpression(self, node: NewExpression) -> str:
        if node.args is not None:
            arg_list_strs = [self.stringify(arg) for arg in node.args.get_children()]
            arg_list = ", ".join(arg_list_strs)
            return f"new {self.stringify(node.type)}({arg_list})"
        else:
            return f"new {self.stringify(node.type)}"

    def stringify_ParenthesizedExpression(self, node: ParenthesizedExpression) -> str:
        return f"({self.stringify(node.expr)})"

    def stringify_PointerExpression(self, node: PointerExpression) -> str:
        return f"{node.op.value}{self.stringify(node.operand)}"

    def stringify_ScopeResolution(self, node: ScopeResolution) -> str:
        if node.scope is None:
            return "::"
        return f"{self.stringify(node.scope)}::"

    def stringify_QualifiedIdentifier(self, node: QualifiedIdentifier) -> str:
        return f"{self.stringify(node.scope)}{self.stringify(node.name)}"

    def stringify_SizeofExpression(self, node: SizeofExpression) -> str:
        if is_expression(node.operand):
            return f"sizeof {self.stringify(node.operand)}"
        else:
            return f"sizeof({self.stringify(node.operand)})"

    def stringify_TernaryExpression(self, node: TernaryExpression) -> str:
        return (
            f"{self.stringify(node.condition)} ? "
            f"{self.stringify(node.consequence)} : "
            f"{self.stringify(node.alternative)}"
        )

    def stringify_ThisExpression(self, node: ThisExpression) -> str:
        return "this"

    def stringify_UnaryExpression(self, node: UnaryExpression) -> str:
        return f"{node.op.value}{self.stringify(node.operand)}"

    def stringify_UpdateExpression(self, node: UpdateExpression) -> str:
        if node.prefix:
            return f"{node.op.value}{self.stringify(node.operand)}"
        else:
            return f"{self.stringify(node.operand)}{node.op.value}"

    def stringify_Modifier(self, node: Modifier) -> str:
        return node.modifier

    def stringify_ModifierList(self, node: ModifierList) -> str:
        return " ".join([self.stringify(mod) for mod in node.get_children()])

    def stringify_AssertStatement(self, node: AssertStatement) -> str:
        if node.message is not None:
            return (
                f"assert {self.stringify(node.condition)} : "
                f"{self.stringify(node.message)};"
            )
        else:
            return f"assert {self.stringify(node.condition)};"

    def stringify_BlockStatement(self, node: BlockStatement) -> str:
        stmt_strs = [self.stringify(stmt) for stmt in node.stmts.get_children()]
        return "{\n" + "\n".join(stmt_strs) + "\n}"

    def stringify_BreakStatement(self, node: BreakStatement) -> str:
        if node.label is not None:
            return f"break {self.stringify(node.label)};"
        else:
            return "break;"

    def stringify_ContinueStatement(self, node: ContinueStatement) -> str:
        if node.label is not None:
            return f"continue {self.stringify(node.label)};"
        else:
            return "continue;"

    def stringify_DoStatement(self, node: DoStatement) -> str:
        return (
            f"do\n{self.stringify(node.body)}\n"
            f"while ({self.stringify(node.condition)});"
        )

    def stringify_EmptyStatement(self, node: EmptyStatement) -> str:
        return ";"

    def stringify_ExpressionStatement(self, node: ExpressionStatement) -> str:
        return f"{self.stringify(node.expr)};"

    def stringify_ForInStatement(self, node: ForInStatement) -> str:
        decl_type_str = self.stringify(node.decl_type)
        iter_str = self.stringify(node.declarator)
        iterable_str = self.stringify(node.iterable)
        body_str = self.stringify(node.body)
        op = node.forin_type.value
        return f"for ({decl_type_str} {iter_str} {op} {iterable_str}) {body_str}"

    def stringify_ForStatement(self, node: ForStatement) -> str:
        if node.init is None:
            init_str = ";"
        elif node.is_init_decl:
            init_str = self.stringify(node.init)
        else:
            init_str = (
                ", ".join(self.stringify(init) for init in node.init.get_children())
                + ";"
            )

        if node.condition is None:
            cond_str = ";"
        else:
            cond_str = " " + self.stringify(node.condition) + ";"

        if node.update is None:
            update_str = ""
        else:
            update_str = " " + ", ".join(
                self.stringify(u) for u in node.update.get_children()
            )

        body_str = self.stringify(node.body)
        return f"for ({init_str}{cond_str}{update_str}) {body_str}"

    def stringify_GotoStatement(self, node: GotoStatement) -> str:
        return f"goto {self.stringify(node.label)};"

    def stringify_IfStatement(self, node: IfStatement) -> str:
        cond_str = self.stringify(node.condition)
        then_str = self.stringify(node.consequence)
        if node.alternate is None:
            return f"if ({cond_str}) {then_str}"
        else:
            else_str = self.stringify(node.alternate)
            return f"if ({cond_str}) {then_str} else {else_str}"

    def stringify_LabeledStatement(self, node: LabeledStatement) -> str:
        return f"{self.stringify(node.label)}: {self.stringify(node.stmt)}"

    def stringify_LambdaExpression(self, node: LambdaExpression) -> str:
        params_str = ", ".join(
            [self.stringify(param) for param in node.params.get_children()]
        )
        body_str = self.stringify(node.body)

        if not node.parenthesized:
            return f"{params_str} -> {body_str}"
        else:
            return f"({params_str}) -> {body_str}"

    def stringify_ReturnStatement(self, node: ReturnStatement) -> str:
        if node.expr is None:
            return "return;"
        else:
            return f"return {self.stringify(node.expr)};"

    def stringify_SwitchCase(self, node: SwitchCase) -> str:
        if node.case is not None:
            case_str = f"case {self.stringify(node.case)}:"
        else:
            case_str = "default:"
        stmts_str = "\n".join(
            [self.stringify(stmt) for stmt in node.stmts.get_children()]
        )
        return f"{case_str}\n{stmts_str}"

    def stringify_SwitchStatement(self, node: SwitchStatement) -> str:
        cond_str = self.stringify(node.condition)
        cases_str = "\n".join(
            [self.stringify(case) for case in node.cases.get_children()]
        )
        return f"switch ({cond_str}) {{\n{cases_str}\n}}"

    def stringify_SynchronizedStatement(self, node: SynchronizedStatement) -> str:
        expr_str = self.stringify(node.expr)
        body_str = self.stringify(node.body)
        return f"synchronized {expr_str} {body_str}"

    def stringify_ThrowStatement(self, node: ThrowStatement) -> str:
        return f"throw {self.stringify(node.expr)};"

    def stringify_CatchClause(self, node: CatchClause) -> str:
        assert node.exception is not None
        assert node.catch_types is not None

        catch_types_str = " | ".join(
            [self.stringify(t) for t in node.catch_types.get_children()]
        )
        exception_str = self.stringify(node.exception)
        body_str = self.stringify(node.body)

        if node.modifiers is not None:
            modifiers_str = self.stringify(node.modifiers)
            return (
                f"catch ({modifiers_str} {catch_types_str} {exception_str}) {body_str}"
            )
        else:
            return f"catch ({catch_types_str} {exception_str}) {body_str}"

    def stringify_FinallyClause(self, node: FinallyClause) -> str:
        body_str = self.stringify(node.body)
        return f"finally {body_str}"

    def stringify_TryStatement(self, node: TryStatement) -> str:
        assert node.handlers is not None

        body_str = self.stringify(node.body)
        handlers_str = "\n".join(
            [self.stringify(handler) for handler in node.handlers.get_children()]
        )
        if node.finalizer is not None:
            finalizer_str = self.stringify(node.finalizer)
            return f"try {body_str}\n{handlers_str}\n{finalizer_str}"
        else:
            return f"try {body_str}\n{handlers_str}"

    def stringify_TryResource(self, node: TryResource) -> str:
        resource_str = self.stringify(node.resource)
        if is_expression(node.resource):
            return resource_str
        else:
            return resource_str[:-1]  # remove semicolon

    def stringify_TryWithResourcesStatement(
        self, node: TryWithResourcesStatement
    ) -> str:
        resource_str = "; ".join(
            self.stringify(res) for res in node.resources.get_children()
        )
        body_str = self.stringify(node.body)

        res_str = f"try ({resource_str}) {body_str}"

        if node.handlers is not None:
            handlers_str = "\n".join(
                [self.stringify(handler) for handler in node.handlers.get_children()]
            )
            res_str += f"\n{handlers_str}"

        if node.finalizer is not None:
            finalizer_str = self.stringify(node.finalizer)
            res_str += f"\n{finalizer_str}"

        return res_str

    def stringify_WhileStatement(self, node: WhileStatement) -> str:
        cond_str = self.stringify(node.condition)
        body_str = self.stringify(node.body)
        return f"while ({cond_str}) {body_str}"

    def stringify_YieldStatement(self, node: YieldStatement) -> str:
        return f"yield {self.stringify(node.expr)};"

    def stringify_DimensionSpecifier(self, node: DimensionSpecifier) -> str:
        if node.expr is not None:
            return f"[{self.stringify(node.expr)}]"
        else:
            return "[]"

    def stringify_Dimensions(self, node: Dimensions) -> str:
        return "".join([self.stringify(dim) for dim in node.get_children()])

    def stringify_TypeIdentifier(self, node: TypeIdentifier) -> str:
        if node.dimension is not None:
            dim_str = self.stringify(node.dimension)
            return f"{node.type_identifier}{dim_str}"
        else:
            return node.type_identifier

    def stringify_TypeParameter(self, node: TypeParameter) -> str:
        if node.extends is not None:
            bounds_str = " & ".join(
                [self.stringify(type_id) for type_id in node.extends.get_children()]
            )
            return f"{node.type_identifier} extends {bounds_str}"
        else:
            return node.type_identifier

    def stringify_TypeParameterList(self, node: TypeParameterList) -> str:
        body_str = ", ".join(
            [self.stringify(type_param) for type_param in node.get_children()]
        )
        return f"<{body_str}>"

    def stringify_InitializingDeclarator(self, node: InitializingDeclarator) -> str:
        if is_expression(node.value):
            return f"{self.stringify(node.declarator)} = {self.stringify(node.value)}"
        else:
            arg_list = ", ".join(
                [self.stringify(arg) for arg in node.value.get_children()]
            )
            return f"{self.stringify(node.declarator)}({arg_list})"

    def stringify_VariableDeclarator(self, node: VariableDeclarator) -> str:
        return self.stringify(node.decl_id)

    def stringify_PointerDeclarator(self, node: PointerDeclarator) -> str:
        return f"*{self.stringify(node.declarator)}"

    def stringify_ReferenceDeclarator(self, node: ReferenceDeclarator) -> str:
        return f'{"&" if not node.r_ref else "&&"}{self.stringify(node.declarator)}'

    def stringify_ArrayDeclarator(self, node: ArrayDeclarator) -> str:
        return f"{self.stringify(node.declarator)}{self.stringify(node.dim)}"

    def stringify_VariadicParameter(self, node: VariadicParameter) -> str:
        return "..."

    def stringify_UntypedParameter(self, node: UntypedParameter) -> str:
        return self.stringify(node.declarator)

    def stringify_TypedFormalParameter(self, node: TypedFormalParameter) -> str:
        if node.declarator is not None:
            return f"{self.stringify(node.decl_type)} {self.stringify(node.declarator)}"
        else:
            return self.stringify(node.decl_type)

    def stringify_SpreadParameter(self, node: SpreadParameter) -> str:
        return f"{self.stringify(node.decl_type)} ...{self.stringify(node.declarator)}"

    def stringify_FunctionDeclarator(self, node: FunctionDeclarator) -> str:
        params_str = ", ".join(
            [self.stringify(param) for param in node.parameters.get_children()]
        )
        return f"{self.stringify(node.declarator)}({params_str})"

    def stringify_FunctionHeader(self, node: FunctionHeader) -> str:
        if node.return_type is None:
            ret_type_str = ""
        else:
            ret_type_str = self.stringify(node.return_type)
        decl_str = self.stringify(node.func_decl)
        res = f"{ret_type_str} {decl_str}"

        if node.type_params is not None:
            type_param_str = self.stringify(node.type_params)
            res = f"{type_param_str} {res}"

        if node.modifiers is not None:
            modifiers_str = self.stringify(node.modifiers)
            res = f"{modifiers_str} {res}"

        if node.throws is not None:
            throws_str = ", ".join(
                self.stringify(throw) for throw in node.throws.get_children()
            )
            res = f"{res} throws {throws_str}"

        return res

    def stringify_FunctionDeclaration(self, node: FunctionDeclaration) -> str:
        header_str = self.stringify(node.header)
        body_str = self.stringify(node.body)
        return f"{header_str} {body_str}"

    def stringify_DeclaratorType(self, node: DeclaratorType) -> str:
        if node.prefix_modifiers is not None:
            prefix_str = self.stringify(node.prefix_modifiers) + " "
        else:
            prefix_str = ""

        if node.postfix_modifiers is not None:
            postfix_str = " " + self.stringify(node.postfix_modifiers)
        else:
            postfix_str = ""

        return f"{prefix_str}{self.stringify(node.type_id)}{postfix_str}"

    def stringify_LocalVariableDeclaration(self, node: LocalVariableDeclaration) -> str:
        decl_strs = ", ".join(
            self.stringify(decl) for decl in node.declarators.get_children()
        )
        res = f"{self.stringify(node.type)} {decl_strs};"
        return res

    def stringify_TypeParameterDeclaration(self, node: TypeParameterDeclaration) -> str:
        return (
            f"{self.stringify(node.typename_opt.value)} {self.stringify(node.type_id)}"
        )

    def stringify_TemplateParameterList(self, node: TemplateParameterList) -> str:
        param_str = ", ".join([self.stringify(param) for param in node.node_list])
        return f"<{param_str}>"

    def stringify_TemplateDeclaration(self, node: TemplateDeclaration) -> str:
        params_str = self.stringify(node.params)
        func_decl_str = self.stringify(node.func_decl)
        return f"template {params_str}\n{func_decl_str}"

    def stringify_AnonymousDeclarator(self, node: AnonymousDeclarator) -> str:
        return ""

    def stringify_SpreadElement(self, node: SpreadElement) -> str:
        expr_str = self.stringify(node.expr)
        return f"...{expr_str}"

    def stringify_AwaitExpression(self, node: AwaitExpression) -> str:
        expr_str = self.stringify(node.expr)
        return f"await {expr_str}"

    def stringify_WithStatement(self, node: WithStatement) -> str:
        obj_str = self.stringify(node.object)
        body_str = self.stringify(node.body)
        return f"with ({obj_str}) {body_str}"

    def stringify_KeyValuePair(self, node: KeyValuePair) -> str:
        key_str = self.stringify(node.key)
        value_str = self.stringify(node.value)

        return f"{key_str}: {value_str}"

    def _stringify_ObjectMethod(self, node: FunctionDeclaration) -> str:
        header_str = self.stringify(node.header)
        body_str = self.stringify(node.body)
        return f"{header_str} {body_str}"

    def stringify_Object(self, node: Object) -> str:
        member_strs = []
        for member in node.members.get_children():
            if isinstance(member, FunctionDeclaration):
                member_strs.append(self._stringify_ObjectMethod(member))
            else:
                member_strs.append(self.stringify(member))

        member_str = ",\n".join(member_strs)
        return f"{{\n{member_str}\n}}"

    def stringify_DestructuringDeclarator(self, node: DestructuringDeclarator) -> str:
        pattern_str = self.stringify(node.pattern)
        return pattern_str

    def stringify_ComputedPropertyName(self, node: ComputedPropertyName) -> str:
        expr_str = self.stringify(node.expr)
        return f"[{expr_str}]"
