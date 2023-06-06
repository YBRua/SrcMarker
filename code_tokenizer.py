import re
import sys
import warnings

from sctokenizer import CppTokenizer, JavaTokenizer
from sctokenizer.token import TokenType, Token
from typing import List


def _join_by_rows(tokens: List[Token]):
    rows = []
    current_row = 0
    for token in tokens:
        if token.line > current_row:
            current_row = token.line
            rows.append([])
        rows[-1].append(token.token_value)

    row_strs = [' '.join(row) for row in rows]
    return row_strs


def fix_join_artifacts(text: str):
    # remove spaces between dots
    text = re.sub(r'\s?\.\s?(?=\w+)', '.', text)
    # remove spaces between underscores
    text = re.sub(r'_\s?(?=\w+)', '_', text)
    # replace 0X with 0x
    text = re.sub(r'0X', '0x', text)
    return text


def fix_single_quotes(input_str: str):
    # removes all spaces between single quotes to fix char pasing
    return re.sub(r"\s+(?=(?:(?:[^']*'){2})*[^']*'[^']*$)", '', input_str)


def tokens_to_strings(tokens: List[Token]):
    row_joined = _join_by_rows(tokens)
    return ' '.join(fix_single_quotes(fix_join_artifacts(x)) for x in row_joined)


def sanitize_name(name):
    # https://github.com/eliphatfs/torch.redstone
    return re.sub(r'\W|^(?=\d)', '_', name)


def _split_name(c_token: str) -> List[str]:
    res = []
    snake_splitted = _try_split_snake(c_token)
    for tok in snake_splitted:
        res.extend(_try_split_camel(tok))
    return res


def _try_split_snake(c_token: str) -> List[str]:
    words = c_token.split('_')
    res = ['_'] * (len(words) * 2 - 1)
    res[0::2] = words
    return res


def _try_split_camel(c_token: str) -> List[str]:
    return re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', c_token).split()


def split_string_literal(c_token: str) -> List[str]:
    # remove escape sequences
    stripped = re.sub(r'\\.', '', c_token)

    return stripped.strip().split()


def split_identifier(c_token: str) -> List[str]:
    if '/' in c_token:  # include path
        res = []
        for subtok in c_token.split('/'):
            res.extend(split_identifier(subtok))
            res.append('/')
        return res
    else:
        return _split_name(c_token)


class CodeTokenizer:
    def __init__(self, lang: str = 'c'):
        self.lang = lang
        if lang in ('c', 'cpp'):
            self.tokenizer = CppTokenizer()
        elif lang == 'java':
            self.tokenizer = JavaTokenizer()
        else:
            raise ValueError('Language must be either "c" or "java"')

    def _tokens_postprocess(self, tokens: List[Token]):
        res = []
        for token in tokens:
            if token.token_type == TokenType.COMMENT_SYMBOL:
                # res.append('__comment__')
                # raise RuntimeError('No comment allowed!')
                # warnings.warn('sctokenizer found "comments" in the code')
                # NOTE: for some reason sctokenizer may create "comments" in the code
                continue
            if token.token_type == TokenType.STRING:
                # res.extend(split_string_literal(token.token_value))
                res.append('__string__')
            elif token.token_type == TokenType.CONSTANT:
                res.append('__constant__')
            elif token.token_type == TokenType.IDENTIFIER:
                res.extend(split_identifier(token.token_value))
            elif len(token.token_value) > 40:
                # the tokenizer is sometimes buggy
                # skip extremely long 'token's
                res.append('<unk>')
            else:
                res.append(token.token_value)
        return res

    def get_tokens(self, source: str):
        code_tokens = self.tokenizer.tokenize(source)
        return code_tokens, self._tokens_postprocess(code_tokens)


if __name__ == '__main__':

    def main(argv):
        """Driver mostly for testing purposes."""
        for filename in argv[1:]:
            with open(filename, 'r') as f:
                source = f.read()
                if source is None:
                    continue
                for token in CodeTokenizer().get_tokens(source):
                    print(token)

    main(sys.argv)
