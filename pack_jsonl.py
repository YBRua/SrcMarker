import os
import json
from tree_sitter import Parser, Language

if __name__ == "__main__":
    DATASET = "./data/gcj_java_funcs"
    JSONL_DATASET_NAME = f"{os.path.basename(DATASET)}.jsonl"

    lang = "java"
    language = Language("./metrics/parser/languages.so", lang)
    parser = Parser()
    parser.set_language(language)

    total_files = 0
    valid_files = 0
    invalid_files = 0

    with open(JSONL_DATASET_NAME, "w") as fo:
        for author in os.listdir(DATASET):
            for file in os.listdir(os.path.join(DATASET, author)):
                with open(os.path.join(DATASET, author, file), "r") as fi:
                    code = fi.read()
                    if lang == "java":
                        wrapped = f"public class Wrapper {{\n{code}\n}}"
                    else:
                        wrapped = code
                    # check tree validity
                    tree = parser.parse(bytes(wrapped, "utf-8"))
                    if tree.root_node.has_error:
                        invalid_files += 1
                        total_files += 1
                        continue

                    # write to jsonl
                    data_dict = {
                        "author": author,
                        "file": file,
                        "original_string": code,
                    }
                    fo.write(json.dumps(data_dict) + "\n")

                    valid_files += 1
                    total_files += 1

    print(f"valid files: {valid_files}")
    print(f"invalid files: {invalid_files}")
    print(f"total files: {total_files}")
