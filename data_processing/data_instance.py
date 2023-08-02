from typing import List, Optional, Tuple
from sctokenizer import Token


class WMDetectionDataInstance:
    def __init__(self, id: str, source: str, tokens: List[str], label: bool):
        self.id = id
        self.source = source
        self.tokens = tokens
        self.label = label

    def __repr__(self) -> str:
        return (f'WMDetectionDataInstance(id={self.id}, label={self.label})')


class DewatermarkingDataInstance:
    def __init__(self, id: str, source: str, source_tokens: List[str], target: str,
                 target_tokens: List[str]):
        self.id = id
        self.source = source
        self.source_tokens = source_tokens
        self.target = target
        self.target_tokens = target_tokens

    def __repr__(self) -> str:
        return f'DewatermarkingDataInstance(id={self.id})'


class DataInstance:
    def __init__(self,
                 id: str,
                 source: str,
                 source_tokens: List[Token],
                 tokens: List[str],
                 task_label: Optional[str] = None,
                 transform_keys: Optional[Tuple[str]] = None) -> None:
        self.id = id
        self.source = source
        self.source_tokens = source_tokens
        self.tokens = tokens
        self.transform_keys = transform_keys
        self.task_label = task_label

    def __repr__(self) -> str:
        return (f'DataInstance(id={self.id}, ' f'transform_keys={self.transform_keys})')
