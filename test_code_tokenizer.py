from code_tokenizer import CodeTokenizer, tokens_to_strings

CODE = """int count_tidy(int _N) {
  const int newnew = 1;
}

"""

if __name__ == '__main__':
    LANG = 'c'
    tokenizer = CodeTokenizer(LANG)
    tokens = tokenizer.tokenizer.tokenize(CODE)
    print(tokens)
    print(tokenizer._tokens_postprocess(tokens))

    print(repr(tokens_to_strings(tokens)))
