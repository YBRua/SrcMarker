import abc
from ..nodes import Node
from ..tree_manip.visitors import TransformingVisitor
from typing import Dict


class CodeTransformer:
    name: str

    @abc.abstractmethod
    def get_available_transforms(self):
        pass

    @abc.abstractmethod
    def code_transform(self, code: str, dst_style: str):
        pass

    @abc.abstractmethod
    def mutable_tree_transform(self, node: Node, dst_style: str):
        pass

    def throw_invalid_dst_style(self, dst_style: str):
        msg = f"invalid dst_style: {dst_style} for {self.__class__.__name__}"
        raise ValueError(msg)
