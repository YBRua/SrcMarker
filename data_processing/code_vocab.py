import json
from collections import defaultdict
from code_tokenizer import sanitize_name

from typing import List


class CodeVocab:
    def __init__(self) -> None:
        self.word2idx = {}
        self.idx2word = []
        self.word_freq = defaultdict(int)
        self.add_word('<pad>')  # 0
        self.add_word('<unk>')  # 1

    def __len__(self):
        return len(self.idx2word)

    def has_word(self, word: str) -> bool:
        return word in self.word2idx

    def _get_latest_idx(self) -> int:
        return len(self.idx2word) - 1

    def add_word(self, word: str) -> int:
        if not self.has_word(word):
            self.idx2word.append(word)
            self.word2idx[word] = self._get_latest_idx()
        self.word_freq[word] += 1
        return self.word2idx[word]

    def filter_vocab_by_frequency(self, min_freq: int = 1):
        filtered = CodeVocab()
        for word, freq in self.word_freq.items():
            if freq >= min_freq:
                filtered.add_word(word)
        return filtered

    def get_id_by_token(self, token: str) -> int:
        return self.word2idx.get(token, self.word2idx['<unk>'])

    def get_token_by_id(self, id: int) -> str:
        return self.idx2word[id]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.get_id_by_token(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.get_token_by_id(id) for id in ids]

    def get_valid_identifier_mask(self):
        mask = []
        for word in self.idx2word:
            if sanitize_name(word) == word:
                # valid identifier, should not mask
                mask.append(0)
            else:
                mask.append(1)

        return mask

    def dump(self, json_path: str = 'code_vocab.json'):
        serialized = {
            'idx2word': self.idx2word,
            'word2idx': self.word2idx,
            'word_freq': self.word_freq
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=4, ensure_ascii=False)

    def load(self, json_path: str = 'code_vocab.json'):
        with open(json_path, 'r', encoding='utf-8') as f:
            serialized = json.load(f)
        self.idx2word = serialized['idx2word']
        self.word2idx = serialized['word2idx']
        self.word_freq = serialized['word_freq']

        return self
