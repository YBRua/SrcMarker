from .literal import Literal
from .identifier import Identifier
from .this import ThisExpression
from .new_expr import NewExpression
from .call_expr import CallExpression
from .field_access import FieldAccess
from .array_access import ArrayAccess
from typing import Union

PrimaryExpression = Union[
    Literal,
    Identifier,
    ThisExpression,
    NewExpression,
    CallExpression,
    FieldAccess,
    ArrayAccess,
]
