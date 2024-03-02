import json
from collections import defaultdict
from code_tokenizer import sanitize_name

from typing import List

JAVA_KEYWORDS = [
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
    "sealed",
]

C_KEYWORDS = [
    "auto",
    "break",
    "bool",
    "case",
    "char",
    "const",
    "continue",
    "class",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "if",
    "elif",
    "else",
    "endif",
    "ifdef",
    "ifndef",
    "define",
    "undef",
    "include",
    "line",
    "error",
    "pragma",
    "defined",
    "__has_c_attribute",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "using",
    "void",
    "volatile",
    "while",
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_Bool",
    "_Complex",
    "_Decimal128",
    "_Decimal32",
    "_Decimal64",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local",
    "and",
    "or",
    "not",
]

ALL_KEYWORDS_SET = set(JAVA_KEYWORDS + C_KEYWORDS)
ALL_KEYWORDS_LOWER = set([kw.lower() for kw in ALL_KEYWORDS_SET])


class CodeVocab:
    def __init__(self) -> None:
        self.word2idx = {}
        self.idx2word = []
        self.word_freq = defaultdict(int)
        self.add_word("<pad>")  # 0
        self.add_word("<mask>")  # 1
        self.add_word("<unk>")  # 2
        self.add_word("<s>")  # 3
        self.add_word("</s>")  # 4

    def pad_idx(self):
        return self.word2idx["<pad>"]

    def mask_idx(self):
        return self.word2idx["<mask>"]

    def bos_idx(self):
        return self.word2idx["<s>"]

    def eos_idx(self):
        return self.word2idx["</s>"]

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

    def get_high_frequency_words(self, top: int = 512):
        most_frequents = list(
            sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        )
        res = [word for word, _ in most_frequents[:top]]
        print(res[-1], self.word_freq[res[-1]])
        return res

    def get_id_by_token(self, token: str) -> int:
        return self.word2idx.get(token, self.word2idx["<unk>"])

    def get_token_by_id(self, id: int) -> str:
        return self.idx2word[id]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.get_id_by_token(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.get_token_by_id(id) for id in ids]

    def _is_reserved(self, word: str):
        word = sanitize_name(word).lower()
        return word in ALL_KEYWORDS_LOWER

    def get_valid_highfreq_mask(self, top: int = 512):
        mask = []
        high_freq_words = self.get_high_frequency_words(top)
        print(high_freq_words)
        for word in self.idx2word:
            if sanitize_name(word) == word and word in high_freq_words:
                # valid high frequency identifier, should not mask
                mask.append(0)
            else:
                mask.append(1)

        return mask

    def get_valid_identifier_mask(self):
        mask = []
        for word in self.idx2word:
            if sanitize_name(word) == word and not self._is_reserved(word):
                # valid identifier, should not mask
                mask.append(0)
            else:
                mask.append(1)

        return mask

    def get_valid_identifier_idx(self):
        idx = []
        for i, word in enumerate(self.idx2word):
            if sanitize_name(word) == word and not self._is_reserved(word):
                idx.append(i)

        return idx

    def dump(self, json_path: str = "code_vocab.json"):
        serialized = {
            "idx2word": self.idx2word,
            "word2idx": self.word2idx,
            "word_freq": self.word_freq,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=4, ensure_ascii=False)

    def load(self, json_path: str = "code_vocab.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            serialized = json.load(f)
        self.idx2word = serialized["idx2word"]
        self.word2idx = serialized["word2idx"]
        self.word_freq = serialized["word_freq"]

        return self
