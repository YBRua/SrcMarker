from ..node import Node, NodeType


class Expression(Node):
    def __init__(self, node_type: NodeType):
        super().__init__(node_type)


def is_expression(node: Node):
    return (
        isinstance(node, Expression) or node.node_type == NodeType.FUNCTION_DEFINITION
    )


def is_primary_expression(node: Node):
    nt = node.node_type
    return nt in {
        NodeType.LITERAL,
        NodeType.IDENTIFIER,
        NodeType.THIS_EXPR,
        NodeType.PARENTHESIZED_EXPR,
        NodeType.NEW_EXPR,
        NodeType.CALL_EXPR,
        NodeType.FIELD_ACCESS,
        NodeType.ARRAY_ACCESS,
    }
