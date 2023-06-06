from ..node import Node, NodeType
from ..expressions import Expression
from ..expressions import is_expression
from .block_stmt import BlockStatement
from .func_declaration import FormalParameterList
from ..utils import throw_invalid_type
from typing import List, Union


class LambdaExpression(Expression):

    def __init__(self,
                 node_type: NodeType,
                 lambda_params: FormalParameterList,
                 body: Union[Expression, BlockStatement],
                 parenthesized: bool = False):
        super().__init__(node_type)
        self.params = lambda_params
        self.body = body
        self.parenthesized = parenthesized

    def _check_types(self):
        if self.node_type != NodeType.LAMBDA_EXPR:
            throw_invalid_type(self.node_type, self)
        if self.params.node_type != NodeType.FORMAL_PARAMETER_LIST:
            throw_invalid_type(self.params.node_type, self, 'params')
        if (self.body.node_type != NodeType.BLOCK_STMT and not is_expression(self.body)):
            throw_invalid_type(self.body.node_type, self, 'body')

    def to_string(self) -> str:
        params_str = ', '.join(
            [param.to_string() for param in self.params.get_children()])

        body_str = self.body.to_string()
        if not self.parenthesized:
            return f'{params_str} -> {body_str}'
        else:
            return f'({params_str}) -> {body_str}'

    def get_children(self) -> List[Node]:
        return [self.params, self.body]

    def get_children_names(self) -> List[str]:
        return ['params', 'body']
