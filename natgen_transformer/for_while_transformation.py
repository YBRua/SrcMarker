import copy
import os
import re
from typing import List, Union, Tuple

import numpy as np

from .language_processors import JavaAndCPPProcessor, JavascriptProcessor
from .transformation_base import NatGenBaseTransformer

processor_function = {
    "java": [
        JavaAndCPPProcessor.for_to_while_random,
        JavaAndCPPProcessor.while_to_for_random,
    ],
    "c": [
        JavaAndCPPProcessor.for_to_while_random,
        JavaAndCPPProcessor.while_to_for_random,
    ],
    "cpp": [
        JavaAndCPPProcessor.for_to_while_random,
        JavaAndCPPProcessor.while_to_for_random,
    ],
    "javascript": [
        JavascriptProcessor.for_to_while_random,
        JavascriptProcessor.while_to_for_random,
    ],
}


class ForWhileTransformer(NatGenBaseTransformer):
    """
    Change the `for` loops with `while` loops and vice versa.
    """

    def __init__(self, parser_path, language):
        super(ForWhileTransformer, self).__init__(
            parser_path=parser_path, language=language
        )
        self.language = language
        self.transformations = processor_function[language]
        processor_map = {
            "java": self.get_tokens_with_node_type,
            "c": self.get_tokens_with_node_type,
            "cpp": self.get_tokens_with_node_type,
            "javascript": JavascriptProcessor.get_tokens,
        }
        self.final_processor = processor_map[self.language]

    def transform_code(
        self,
        code: Union[str, bytes],
    ) -> Tuple[str, object]:
        success = False
        transform_functions = copy.deepcopy(self.transformations)
        while not success and len(transform_functions) > 0:
            function = np.random.choice(transform_functions)
            transform_functions.remove(function)
            modified_root, modified_code, success = function(code, self)
            if success:
                code = modified_code
        root_node = self.parse_code(code=code)
        return_values = self.final_processor(code=code.encode(), root=root_node)
        if isinstance(return_values, tuple):
            tokens, types = return_values
        else:
            tokens, types = return_values, None
        return (
            re.sub("[ \t\n]+", " ", " ".join(tokens)),
            {"types": types, "success": success},
        )

    def transform_for_to_while(self, code: Union[str, bytes]) -> str:
        function = self.transformations[0]  # for-to-while
        modified_root, modified_code, success = function(code, self)
        return modified_code

    def transform_while_to_for(self, code: Union[str, bytes]) -> str:
        function = self.transformations[1]  # while-to-for
        _, modified_code, _ = function(code, self)
        return modified_code

    def get_available_transforms(self) -> List[str]:
        return ["for_to_while", "while_to_for"]

    def transform(self, code: str, transform: str):
        if transform == "for_to_while":
            return self.transform_for_to_while(code)
        elif transform == "while_to_for":
            return self.transform_while_to_for(code)
        else:
            raise ValueError(f"Unknown transform {transform}")


if __name__ == "__main__":
    java_code = """
    class A{
        int foo(int n){
            int res = 0;
            for(i = 0; i < n; i++) {
                int j = 0;
                if (i == 0){
                    foo(7);
                    continue;
                }
                else if (res == 9) {
                    bar(8);
                    break;
                }
                else if (n == 0){
                    tar(9);
                    return 0;
                }
                else{
                    foo();
                    bar();
                    tar();
                }
            }
            return res;
        }
    }
    """
    input_map = {
        "java": ("java", java_code),
    }
    parser_path = os.path.join("~/code-watermarking/parser/languages.so")
    for lang in ["java"]:
        lang, code = input_map[lang]
        for_while_transformer = ForWhileTransformer(parser_path, lang)
        print(lang, end="\t")
        code = for_while_transformer.transform_for_to_while(code)
        print(code)
