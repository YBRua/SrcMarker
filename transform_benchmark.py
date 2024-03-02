import json
import time
import tree_sitter
from tqdm import trange

from natgen_transformer import ForWhileTransformer
from metrics.syntax_match import check_tree_validity
from code_transform_provider import CodeTransformProvider
import mutable_tree.transformers as ast_transformers
from ropgen_transform import srcml_interface
from ropgen_transform.code_transformers import LoopTransformer
from typing import List


def check_transformed_codes(res: List[str], parser: tree_sitter.Parser):
    tot = 0
    correct = 0
    for r in res:
        # r = f'public class Wrapper {{ {r} }}'
        tree = parser.parse(r.encode())
        if check_tree_validity(tree.root_node):
            correct += 1
        else:
            print(r)
        tot += 1
    return correct, tot


if __name__ == "__main__":
    JSONL_FILE_PATH = "./datasets/github_c_funcs/test.jsonl"
    parser = tree_sitter.Parser()
    lang = tree_sitter.Language("./parser/languages.so", name="cpp")
    parser.set_language(lang)

    res = []
    print("======== MutableAST ========")
    loop_transformer = ast_transformers.LoopTransformer()
    transform_key = [loop_transformer.TRANSFORM_LOOP_WHILE]

    computer = CodeTransformProvider(
        lang="cpp", parser=parser, transformers=[loop_transformer]
    )

    mutableast_start = time.time()
    with open(JSONL_FILE_PATH, "r", encoding="utf-8") as fi:
        json_objs = [json.loads(line) for line in fi.readlines()]

    tot_files = 0
    for json_obj in json_objs:
        tot_files += 1
        code = json_obj["original_string"]
        t_code = computer.code_transform(code, transform_key)
        # t_code = f'public class Wrapper {{ {t_code} }}'
        res.append(t_code)
    mutableast_time = time.time() - mutableast_start

    print(f"transformed {tot_files} files")
    print(f"time: {mutableast_time * 1000:.2f}ms")
    print(f"average time: {mutableast_time / tot_files * 1000:.2f}ms")

    correct, tot = check_transformed_codes(res, parser)
    print(f"correct: {correct}/{tot} ({correct / tot * 100:.2f}%)")

    print("======== Natgen ========")
    natgen_transformer = ForWhileTransformer("./parser/languages.so", "cpp")

    natgen_start = time.time()
    with open(JSONL_FILE_PATH, "r", encoding="utf-8") as fi:
        json_objs = [json.loads(line) for line in fi.readlines()]

    tot_files = 0
    res = []
    for json_obj in json_objs:
        tot_files += 1
        code = json_obj["original_string"]
        # code = f'public class Wrapper {{ {code} }}'
        t_code = natgen_transformer.transform_for_to_while(code)
        res.append(t_code)
    natgen_time = time.time() - natgen_start

    print(f"transformed {tot_files} files")
    print(f"time: {natgen_time * 1000:.2f}ms")
    print(f"average time: {natgen_time / tot_files * 1000:.2f}ms")

    correct, tot = check_transformed_codes(res, parser)
    print(f"correct: {correct}/{tot} ({correct / tot * 100:.2f}%)")

    print("======== RopGen ========")
    BENCHMARK_DIR = "./data/benchmark_funcs"
    with open(JSONL_FILE_PATH, "r", encoding="utf-8") as fi:
        json_objs = [json.loads(line) for line in fi.readlines()]

    for i, json_obj in enumerate(json_objs):
        code = json_obj["original_string"]
        with open(f"{BENCHMARK_DIR}/{i}.c", "w", encoding="utf-8") as fo:
            fo.write(code)

    ropgen_transformer = LoopTransformer()

    ropgen_start = time.time()
    for i in trange(len(json_objs)):
        input_fpath = f"{BENCHMARK_DIR}/{i}.c"
        input_xml_fpath = f"{BENCHMARK_DIR}/{i}.xml"
        output_xml_fpath = f"{BENCHMARK_DIR}/{i}_t.xml"
        output_fpath = f"{BENCHMARK_DIR}/{i}_t.c"

        srcml_interface.srcml_program_to_xml(input_fpath, input_xml_fpath)
        ropgen_transformer.xml_transform_all(input_xml_fpath, "20.2", output_xml_fpath)
        srcml_interface.srcml_xml_to_program(output_xml_fpath, output_fpath)
    ropgen_time = time.time() - ropgen_start

    print(f"transformed {tot_files} files")
    print(f"time: {ropgen_time * 1000:.2f}ms")
    print(f"average time: {ropgen_time / tot_files * 1000:.2f}ms")

    res = []
    for i in trange(len(json_objs)):
        with open(f"{BENCHMARK_DIR}/{i}_t.c", "r", encoding="utf-8") as fi:
            res.append(fi.read())
            # res.append(f'public class Wrapper {{ {fi.read()} }}')

    correct, tot = check_transformed_codes(res, parser)
    print(f"correct: {correct}/{tot} ({correct / tot * 100:.2f}%)")
