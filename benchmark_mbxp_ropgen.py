import sys
import time
import tree_sitter

from tqdm import tqdm
from ropgen_transform import srcml_interface
import ropgen_transform.code_transformers as ropgen
from benchmark_mbxp import (
    MBCPPRunner,
    MBJPRunner,
    MBJSPRunner,
    compose_function_java,
    compose_function_cpp,
    compose_function_javascript,
)
from benchmark_mbxp import load_samples_with_sol


def main(lang: str):
    LANG = lang
    PARSER_PATH = "./parser/languages.so"
    MBXP_NAME = {"java": "mbjp", "cpp": "mbcpp", "javascript": "mbjsp"}[LANG]
    FILE_STORE = "./mbxp/ropgen/source/"
    BIN_STORE = "./mbxp/ropgen/bin/"
    MBXP_DATASET_PATH = f"./datasets/{MBXP_NAME}_release_v1.2_filtered.jsonl"
    EXT = {"java": "java", "cpp": "cpp", "javascript": "js"}[LANG]

    if LANG == "cpp":
        prefix = "#include <bits/stdc++.h>\nusing namespace std;\n"
    elif LANG == "java":
        prefix = (
            "import java.io.*;\nimport java.lang.*;\n"
            "import java.util.*;\nimport java.math.*;\n"
        )
    elif LANG == "javascript":
        prefix = ""
    else:
        raise ValueError(f"Unknown language: {LANG}")

    parser = tree_sitter.Parser()
    lang = tree_sitter.Language("./parser/languages.so", name=LANG)
    parser.set_language(lang)

    samples = load_samples_with_sol(MBXP_DATASET_PATH)
    print(f"Number of samples with solution: {len(samples)}")

    if LANG == "cpp":
        runner = MBCPPRunner(FILE_STORE, BIN_STORE)
    elif LANG == "java":
        runner = MBJPRunner(FILE_STORE, BIN_STORE)
    elif LANG == "javascript":
        runner = MBJSPRunner(FILE_STORE, BIN_STORE)
    else:
        raise RuntimeError("Unreachable")

    BENCHMARK_DIR = "./data/benchmark_funcs"
    transform_times = []
    transform_pass_rates = []
    for transformer in ropgen.get_transformers():
        print(transformer.__class__.__name__)
        for transform_key in transformer.get_available_transforms():
            print("=" * 80)
            print(f"Running {transform_key}...")

            n_samples = 0
            n_successes = 0
            n_fails = 0
            tot_time = 0.0
            transformed_instances = []

            prog = tqdm(samples)
            prog.set_description("Transforming")
            for i, sample in enumerate(prog):
                if LANG == "cpp":
                    function = compose_function_cpp(sample)
                elif LANG == "java":
                    function = compose_function_java(sample)
                elif LANG == "javascript":
                    function = compose_function_javascript(sample)
                else:
                    raise RuntimeError("Unreachable")
                with open(f"{BENCHMARK_DIR}/{i}.{EXT}", "w", encoding="utf-8") as fo:
                    fo.write(function)

                n_samples += 1
                input_fpath = f"{BENCHMARK_DIR}/{i}.{EXT}"
                input_xml_fpath = f"{BENCHMARK_DIR}/{i}.xml"
                output_xml_fpath = f"{BENCHMARK_DIR}/{i}_t.xml"
                output_fpath = f"{BENCHMARK_DIR}/{i}_t.{EXT}"

                t_start = time.time()
                srcml_interface.srcml_program_to_xml(input_fpath, input_xml_fpath)
                transformer.xml_transform_all(
                    input_xml_fpath, transform_key, output_xml_fpath
                )
                srcml_interface.srcml_xml_to_program(output_xml_fpath, output_fpath)
                t_end = time.time()

                tot_time += t_end - t_start

                with open(output_fpath, "r", encoding="utf-8") as fi:
                    t_function = fi.read()

                if LANG == "java":
                    # have to use a nasty hard-coded index to extract the class header
                    class_header = sample["prompt"].strip().split("\n")[6]
                    t_function = f"{class_header}\n{t_function}\n}}"
                t_program = prefix + t_function + sample["test"]

                task_id = sample["task_id"].split("/")[-1]
                transformed_instances.append((task_id, t_program))

            print(f"Number of transformed samples: {len(transformed_instances)}")

            res, failures = runner.check_solutions(
                transformed_instances, n_processes=16
            )
            n_fails += len(failures)
            n_successes = res[runner.msg_good]

            assert n_samples == n_successes + n_fails

            transform_times.append(tot_time / n_samples)
            transform_pass_rates.append(n_successes / n_samples)

            print(f"Number of samples: {n_samples}")
            print(f"Number of successes: {n_successes}")
            print(f"Number of failures: {n_fails}")
            print(f"Average time: {tot_time / n_samples*1000:.2f} ms")
            print(f"Pass rate: {n_successes / n_samples * 100:.2f}%")
            print(res, failures)
            print()
    print("Times:")
    print(transform_times)
    print("Pass rates:")
    print(transform_pass_rates)

    avg_time = sum(transform_times) / len(transform_times)
    avg_pass_rate = sum(transform_pass_rates) / len(transform_pass_rates)
    print(f"Average time: {avg_time*1000:.2f} ms")
    print(f"Average pass rate: {avg_pass_rate*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark_mbxp_natgen.py <lang>")
        exit(0)

    main(lang=sys.argv[1])
