from .node import Node, NodeType
from typing import Optional


def throw_invalid_type(ty: NodeType, obj: Node, attr: Optional[str] = None):
    if attr is not None:
        msg = f"Invalid type: {ty} for {attr} of {type(obj).__name__}"
    else:
        msg = f"Invalid type: {ty} for {type(obj).__name__}"
    raise TypeError(msg)
