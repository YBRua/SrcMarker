import re
import sys
import tree_sitter

from sctokenizer import CppTokenizer, JavaTokenizer
from sctokenizer.token import TokenType, Token
from typing import List


def _join_by_rows(tokens: List[Token]):
    rows = []
    current_row = -1
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

    # remove enclosing quotes
    if stripped.startswith('"') or stripped.startswith("'"):
        stripped = stripped[1:]
    if stripped.endswith('"') or stripped.endswith("'"):
        stripped = stripped[:-1]

    return ['"'] + stripped.strip().split() + ['"']


def split_identifier(c_token: str) -> List[str]:
    if '/' in c_token:  # include path
        res = []
        for subtok in c_token.split('/'):
            res.extend(split_identifier(subtok))
            res.append('/')
        return res
    else:
        return _split_name(c_token)


class JavaScriptTokenizer:
    def __init__(self):
        parser = tree_sitter.Parser()
        parser_lang = tree_sitter.Language('./parser/languages.so', 'javascript')
        parser.set_language(parser_lang)
        self.parser = parser

    def collect_tokens(self, root: tree_sitter.Node) -> List[Token]:
        tokens = []

        def _collect_token(node: tree_sitter.Node):
            if node.type == 'comment':
                return
            elif node.type in {'number'}:
                tokens.append(
                    Token(node.text.decode(), TokenType.CONSTANT, node.start_point[0],
                          node.start_point[1]))
            elif node.type in {'string', 'template_string', 'regex'}:
                tokens.append(
                    Token(node.text.decode(), TokenType.STRING, node.start_point[0],
                          node.start_point[1]))
            elif node.type in {
                    'identifier', 'shorthand_property_identifier',
                    'shorthand_property_identifier_pattern'
            }:
                tokens.append(
                    Token(node.text.decode(), TokenType.IDENTIFIER, node.start_point[0],
                          node.start_point[1]))
            elif node.child_count == 0:
                tokens.append(
                    Token(node.text.decode(), TokenType.KEYWORD, node.start_point[0],
                          node.start_point[1]))
            else:
                assert node.child_count > 0
                for ch in node.children:
                    _collect_token(ch)

        _collect_token(root)
        return tokens

    def tokenize(self, code: str) -> List[Token]:
        tree = self.parser.parse(bytes(code, 'utf-8'))
        tokens = self.collect_tokens(tree.root_node)
        return tokens


class CodeTokenizer:
    def __init__(self, lang: str = 'c'):
        self.lang = lang
        if lang in ('c', 'cpp'):
            self.tokenizer = CppTokenizer()
        elif lang == 'java':
            self.tokenizer = JavaTokenizer()
        elif lang == 'javascript':
            self.tokenizer = JavaScriptTokenizer()
        else:
            raise ValueError(f'Unsupported language: {lang}')

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
                res.extend(split_string_literal(token.token_value))
                # res.append('__string__')
            elif token.token_type == TokenType.CONSTANT:
                res.append(token.token_value)
                # res.append('__constant__')
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
