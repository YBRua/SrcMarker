from ..nodes import Node
from .code_transformer import CodeTransformer
from typing import List, Dict, Sequence


class TransformerPipeline:
    def __init__(self, transformers: List[CodeTransformer]) -> None:
        self.names = [transformer.name for transformer in transformers]
        self.transformers: Dict[str, CodeTransformer] = {
            transformer.name: transformer for transformer in transformers
        }

    def get_transformer_names(self) -> List[str]:
        return self.names

    def get_transformer(self, name: str) -> CodeTransformer:
        return self.transformers[name]

    def mutable_tree_transform(self, node: Node, keys: Sequence[str]) -> Node:
        for key in keys:
            name = key.split(".")[0]
            node = self.transformers[name].mutable_tree_transform(node, key)
        return node
