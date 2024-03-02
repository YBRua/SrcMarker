from enum import Enum
from typing import List


# types
class NodeType(Enum):
    # top-level
    PROGRAM = "Program"

    # expressions
    ASSIGNMENT_EXPR = "AssignmentExpression"
    BINARY_EXPR = "BinaryExpression"
    INSTANCEOF_EXPR = "InstanceofExpression"
    SIZEOF_EXPR = "SizeofExpression"
    LAMBDA_EXPR = "LambdaExpression"
    TERNARY_EXPR = "TernaryExpression"
    UPDATE_EXPR = "UpdateExpression"
    UNARY_EXPR = "UnaryExpression"
    CAST_EXPR = "CastExpression"
    ARRAY_CREATION_EXPR = "ArrayCreationExpression"
    COMMA_EXPR = "CommaExpression"  # NOTE: not quite an expr
    POINTER_EXPR = "PointerExpression"
    DELETE_EXPR = "DeleteExpression"
    COMPOUND_LITERAL_EXPR = "CompoundLiteralExpression"
    SPREAD_ELEMENT = "SpreadElement"
    AWAIT_EXPR = "AwaitExpression"

    # primary expressions
    LITERAL = "Literal"
    IDENTIFIER = "Identifier"
    QUALIFIED_IDENTIFIER = "QualifiedIdentifier"
    ARRAY_EXPR = "ArrayExpression"
    THIS_EXPR = "ThisExpression"
    PARENTHESIZED_EXPR = "ParenthesizedExpression"
    NEW_EXPR = "NewExpression"
    CALL_EXPR = "CallExpression"
    FIELD_ACCESS = "FieldAccess"
    ARRAY_ACCESS = "ArrayAccess"

    # statements
    ASSERT_STMT = "AssertStatement"
    BLOCK_STMT = "BlockStatement"
    BREAK_STMT = "BreakStatement"
    CONTINUE_STMT = "ContinueStatement"
    DECLARATION_STMT = "DeclarationStatement"  # NOT IMPLEMENTED
    DO_STMT = "DoStatement"
    EMPTY_STMT = "EmptyStatement"
    EXPRESSION_STMT = "ExpressionStatement"
    FOR_STMT = "ForStatement"
    FOR_IN_STMT = "ForInStatement"
    IF_STMT = "IfStatement"
    LABELED_STMT = "LabeledStatement"
    LOCAL_VARIABLE_DECLARATION = "LocalVariableDeclaration"
    RETURN_STMT = "ReturnStatement"
    SWITCH_STMT = "SwitchStatement"
    SYNCHRONIZED_STMT = "SynchronizedStatement"
    THROW_STMT = "ThrowStatement"
    TRY_STMT = "TryStatement"
    TRY_WITH_RESOURCES_STMT = "TryWithResourcesStatement"
    WHILE_STMT = "WhileStatement"
    YIELD_STMT = "YieldStatement"
    GOTO_STMT = "GotoStatement"
    WITH_STMT = "WithStatement"

    # declarations & definitions (are also statements)
    FUNCTION_DECLARATOR = "FunctionDeclarator"
    FUNCTION_HEADER = "FunctionHeader"
    FUNCTION_DEFINITION = "FunctionDefinition"
    TEMPLATE_DECLARATION = "TemplateDeclaration"

    DECLARATOR_TYPE = "DeclaratorType"
    VARIABLE_DECLARATOR = "VariableDeclarator"
    POINTER_DECLARATOR = "PointerDeclarator"
    REFERENCE_DECLARATOR = "ReferenceDeclarator"
    ARRAY_DECLARATOR = "ArrayDeclarator"
    INITIALIZING_DECLARATOR = "InitializingDeclarator"
    ANONYMOUS_DECLARATOR = "AnonymousDeclarator"
    DESTRUCTURING_DECLARATOR = "DestructuringDeclarator"

    DECLARATOR_LIST = "DeclaratorList"

    # types
    TYPE_IDENTIFIER = "TypeIdentifier"
    TYPE_IDENTIFIER_LIST = "TypeIdentifierList"
    TYPE_PARAMETER = "TypeParameter"
    TYPE_PARAMETER_LIST = "TypeParameterList"
    TYPE_PARAMETER_DECLARATION = "TypeParameterDeclaration"
    TEMPLATE_PARAMETER_LIST = "TemplateParameterList"
    DIMENSION_SPECIFIER = "DimensionSpecifier"
    DIMENSIONS = "Dimensions"

    # miscs
    SWITCH_CASE = "SwitchCase"
    CATCH_CLAUSE = "CatchClause"
    FINALLY_CLAUSE = "FinallyClause"
    EXPRESSION_LIST = "ExpressionList"
    STATEMENT_LIST = "StatementList"
    SWITCH_CASE_LIST = "SwitchCaseList"
    TRY_HANDLERS = "TryHandlers"
    TRY_RESOURCE = "TryResource"
    TRY_RESOURCE_LIST = "TryResourceList"
    SCOPE_RESOLUTION = "ScopeResolution"

    MODIFIER = "Modifier"
    MODIFIER_LIST = "ModifierList"
    UNTYPED_PARAMETER = "UntypedParameter"
    FORMAL_PARAMETER = "FormalParameter"
    FORMAL_PARAMETER_LIST = "FormalParameterList"
    SPREAD_PARAMETER = "SpreadParameter"
    VARIADIC_PARAMETER = "VariadicParameter"

    # object
    OBJECT = "Object"
    KEYVALUE_PAIR = "KeyValuePair"
    OBJECT_MEMBERS = "ObjectMembers"
    COMPUTED_PROPERTY_NAME = "ComputedPropertyName"


class Node:
    node_type: NodeType

    def __init__(self, node_type: NodeType):
        self.node_type = node_type

    def _check_types(self):
        raise NotImplementedError("Base class Node should never be initialized")

    def get_children(self) -> List["Node"]:
        raise NotImplementedError()

    def get_children_names(self) -> List[str]:
        raise NotImplementedError()

    def get_child_at(self, attr: str) -> "Node":
        return getattr(self, attr)

    def set_child_at(self, attr: str, value: "Node"):
        if not hasattr(self, attr):
            raise AttributeError(
                f"{type(self).__name__} does not have attribute {attr}"
            )
        setattr(self, attr, value)


class NodeList(Node):
    node_list: List[Node]

    def __init__(self, node_type: NodeType):
        super().__init__(node_type)

    def get_children(self) -> List[Node]:
        return self.node_list

    def get_children_names(self) -> List[int]:
        return list(range(len(self.node_list)))

    def get_child_at(self, index: int) -> Node:
        return self.node_list[index]

    def set_child_at(self, index: int, value: Node):
        self.node_list[index] = value

    def replace_child_at(self, index: int, values: List[Node]):
        self.node_list[index : index + 1] = values
