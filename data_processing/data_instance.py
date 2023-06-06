from typing import List, Optional, Tuple
from sctokenizer import Token


class DataInstance:
    def __init__(self,
                 id: str,
                 source: str,
                 source_tokens: List[Token],
                 tokens: List[str],
                 transform_keys: Optional[Tuple[str]] = None) -> None:
        self.id = id
        self.source = source
        self.source_tokens = source_tokens
        self.tokens = tokens
        self.transform_keys = transform_keys

    def __repr__(self) -> str:
        return (f'DataInstance(id={self.id}, ' f'transform_keys={self.transform_keys})')
