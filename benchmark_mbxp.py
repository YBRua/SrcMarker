import os
import sys
import time
import json
import shutil
import subprocess
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import tree_sitter
from dataset_filter import remove_comments
from code_transform_provider import CodeTransformProvider
import mutable_tree.transformers as ast_transformers


class MBXPRunner:
    task_prefix = 'MBXP'
    lang = 'txt'

    def __init__(self, source_dir: str, bin_dir: str):
        self.source_dir = source_dir
        self.bin_dir = bin_dir

        self.msg_err_compile = 'Compilation'
        self.msg_err_exec = 'Execution'
        self.msg_good = 'Good'

        self.check_cleanup()

    def _remove_nonempty_dir(self, dir_path: str):
        if os.path.exists(dir_path) and len(os.listdir(dir_path)) > 0:
            print(f'{dir_path} is not empty, removing...')
            shutil.rmtree(dir_path)

    def check_cleanup(self):
        self._remove_nonempty_dir(self.source_dir)
        self._remove_nonempty_dir(self.bin_dir)

        if not os.path.exists(self.source_dir):
            os.makedirs(self.source_dir)
        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)

    def _exec(self, cmd: List[str], cwd: Optional[str] = None):
        try:
            exec_result = subprocess.run(
                cmd,
                timeout=15,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
            )
            return exec_result.returncode == 0
        except Exception as e:
            print(e)
            return False

    def _compile(self, src_path: str, bin_path: str):
        raise NotImplementedError()

    def _run_ut(self, bin_path: str):
        raise NotImplementedError()

    def _compile_and_check_impl(self, id: str, code: str, src_path: str, bin_path: str):
        compiled = self._compile(src_path, bin_path)
        if not compiled:
            return id, False, self.msg_err_compile + ': ' + src_path

        # run compiled program
        passed = self._run_ut(bin_path)
        if not passed:
            return id, False, self.msg_err_exec + ': ' + src_path

        return id, True, self.msg_good

    def compile_and_check_solution(self, task_id: str, code: str):
        task_name = f'{self.task_prefix}_{task_id}'
        src_path = os.path.join(self.source_dir, f'{task_name}.{self.lang}')
        bin_path = os.path.join(self.bin_dir, f'{task_name}')

        return self._compile_and_check_impl(task_id, code, src_path, bin_path)

    def check_solution_wrapper(self, args):
        return self.compile_and_check_solution(*args)

    def write_code_to_fs(self, instances: List):
        for task_id, code in instances:
            task_name = f'{self.task_prefix}_{task_id}'
            src_path = os.path.join(self.source_dir, f'{task_name}.{self.lang}')
            with open(src_path, 'w', encoding='utf-8') as f:
                f.write(code)

    def check_solutions(self, instances: List, n_processes: int = 1):
        results = []

        self.write_code_to_fs(instances)

        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            completion_ids = Counter()
            n_samples = 0

            for instance in instances:
                futures.append(executor.submit(self.check_solution_wrapper, instance))
                completion_ids[instance[0]] += 1
                n_samples += 1

            for future in tqdm(as_completed(futures), total=n_samples):
                result = future.result()
                results.append(result)

        # results = []
        # for instance in tqdm(instances):
        #     results.append(self.check_solution_wrapper(instance))

        res_dict = defaultdict(int)
        failures = list()
        for (id, is_good, msg) in results:
            res_dict[msg.split(':')[0]] += 1
            if not is_good:
                failures.append((id, msg))

        return res_dict, failures


class MBCPPRunner(MBXPRunner):
    task_prefix = 'mbcpp'
    lang = 'cpp'

    def __init__(self, source_dir: str, bin_dir: str):
        super().__init__(source_dir, bin_dir)

    def _compile(self, src_path: str, bin_path: str):
        compiled = self._exec(['g++', src_path, '-o', bin_path])
        return compiled

    def _run_ut(self, bin_path: str):
        passed = self._exec([bin_path])
        return passed


class MBJPRunner(MBXPRunner):
    JDK_BATH_PATH = '/usr/lib/jvm/java-8-openjdk-amd64/bin'  # NOTE: change this to your JDK path
    task_prefix = 'mbjp'
    lang = 'java'

    def __init__(self, source_dir: str, bin_dir: str):
        super().__init__(source_dir, bin_dir)

    def _compile(self, src_path: str, bin_path: str):
        javac_path = os.path.join(self.JDK_BATH_PATH, 'javac')
        compiled = self._exec([javac_path, src_path])
        return compiled

    def _run_ut(self, bin_path: str):
        java_path = os.path.join(self.JDK_BATH_PATH, 'java')
        passed = self._exec([java_path, '-cp', bin_path, 'Main'])
        return passed

    def write_code_to_fs(self, instances: List):
        for task_id, code in instances:
            task_name = f'{self.task_prefix}_{task_id}'
            src_path = os.path.join(self.source_dir, task_name, f'main.{self.lang}')
            if not os.path.exists(os.path.dirname(src_path)):
                os.makedirs(os.path.dirname(src_path))

            with open(src_path, 'w', encoding='utf-8') as f:
                f.write(code)

    def compile_and_check_solution(self, task_id: str, code: str):
        task_name = f'{self.task_prefix}_{task_id}'
        src_path = os.path.join(self.source_dir, task_name, f'main.{self.lang}')
        bin_path = os.path.dirname(src_path)

        if not os.path.exists(bin_path):
            os.makedirs(bin_path)

        return self._compile_and_check_impl(task_id, code, src_path, bin_path)


class MBJSPRunner(MBXPRunner):
    task_prefix = 'mbjsp'
    lang = 'js'

    def __init__(self, source_dir: str, bin_dir: str):
        super().__init__(source_dir, bin_dir)

    def check_cleanup(self):
        super().check_cleanup()

    def _compile(self, src_path: str, bin_path: str):
        return True

    def _run_ut(self, bin_path: str):
        passed = self._exec(['node', bin_path])
        return passed

    def compile_and_check_solution(self, task_id: str, code: str):
        task_name = f'{self.task_prefix}_{task_id}'
        src_path = os.path.join(self.source_dir, f'{task_name}.{self.lang}')

        return self._compile_and_check_impl(task_id, code, src_path, src_path)

    def check_solutions(self, instances: List, n_processes: int = 1):
        results = []

        self.write_code_to_fs(instances)
        self._exec(['npm', 'install', 'lodash'], cwd=self.source_dir)

        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            completion_ids = Counter()
            n_samples = 0

            for instance in instances:
                futures.append(executor.submit(self.check_solution_wrapper, instance))
                completion_ids[instance[0]] += 1
                n_samples += 1

            for future in tqdm(as_completed(futures), total=n_samples):
                result = future.result()
                results.append(result)

        # results = []
        # for instance in tqdm(instances):
        #     results.append(self.check_solution_wrapper(instance))

        res_dict = defaultdict(int)
        failures = set()
        for (id, is_good, msg) in results:
            res_dict[msg.split(':')[0]] += 1
            if not is_good:
                failures.add((id, msg))

        return res_dict, failures


def load_samples_with_sol(dataset_path: str):
    with open(dataset_path) as f:
        samples = [json.loads(line) for line in f.readlines()]

    return list(filter(lambda x: x['canonical_solution'] is not None, samples))


def compose_function(sample: Dict) -> str:
    return sample["prompt"] + sample["canonical_solution"]


def compose_function_java(sample: Dict) -> str:
    # remove the closing } for class definition
    sol = sample["canonical_solution"]
    sol = sol.strip().split('\n')[:-1]
    sol = '\n'.join(sol)
    func = sample["prompt"].strip().split('\n')[-1] + '\n' + sol
    return remove_comments(func)


def compose_function_cpp(sample: Dict) -> str:
    func = sample["prompt"].strip().split('\n')[-1] + '\n' + sample["canonical_solution"]
    return remove_comments(func)


def compose_program(sample: Dict) -> str:
    return sample["prompt"] + sample["canonical_solution"] + sample["test"]


def compose_function_javascript(sample: Dict) -> str:
    func = sample["prompt"].strip().split('\n')[-1] + '\n' + sample["canonical_solution"]
    return remove_comments(func)


def main(lang: str):
    LANG = lang
    MBXP_NAME = {'java': 'mbjp', 'cpp': 'mbcpp', 'javascript': 'mbjsp'}[LANG]
    FILE_STORE = './mbxp/original/source/'
    BIN_STORE = './mbxp/original/bin/'
    MBXP_DATASET_PATH = f'./datasets/{MBXP_NAME}_release_v1.2_filtered.jsonl'

    if LANG == 'cpp':
        prefix = "#include <bits/stdc++.h>\nusing namespace std;\n"
    elif LANG == 'java':
        prefix = ('import java.io.*;\nimport java.lang.*;\n'
                  'import java.util.*;\nimport java.math.*;\n')
    elif LANG == 'javascript':
        prefix = ''
    else:
        raise ValueError(f'Unknown language: {LANG}')

    parser = tree_sitter.Parser()
    lang = tree_sitter.Language('./parser/languages.so', name=LANG)
    parser.set_language(lang)

    samples_with_sol = load_samples_with_sol(MBXP_DATASET_PATH)
    print(f'Number of samples with solution: {len(samples_with_sol)}')

    valid_samples = []
    for sample in samples_with_sol:
        function = compose_program(sample)
        tree = parser.parse(function.encode('utf-8'))
        if tree.root_node.has_error:
            continue
        valid_samples.append(sample)
    print(f'Number of valid samples: {len(valid_samples)}')

    if LANG == 'cpp':
        runner = MBCPPRunner(FILE_STORE, BIN_STORE)
    elif LANG == 'java':
        runner = MBJPRunner(FILE_STORE, BIN_STORE)
    elif LANG == 'javascript':
        runner = MBJSPRunner(FILE_STORE, BIN_STORE)
    else:
        raise RuntimeError('Unreachable')

    transform_times = []
    transform_pass_rates = []
    for transformer in ast_transformers.get_all_transformers():
        # for transformer in [ast_transformers.LoopTransformer()]:
        print(transformer.get_available_transforms())
        for transform_key in transformer.get_available_transforms():
            # for transform_key in [transformer.TRANSFORM_LOOP_WHILE]:
            print('=' * 80)
            print(f'Running {transform_key}...')

            computer = CodeTransformProvider(lang=LANG,
                                             parser=parser,
                                             transformers=[transformer])

            n_samples = 0
            n_successes = 0
            n_fails = 0
            tot_time = 0.
            mutableast_instances = []
            for sample in valid_samples:
                n_samples += 1
                if LANG == 'cpp':
                    function = compose_function_cpp(sample)
                elif LANG == 'java':
                    function = compose_function_java(sample)
                elif LANG == 'javascript':
                    function = compose_function_javascript(sample)
                else:
                    raise RuntimeError('Unreachable')
                # t_function = computer.code_transform(function, [transform_key])

                transform_good = True
                t_start = time.time()
                try:
                    t_function = computer.code_transform(function, [transform_key])
                except Exception as e:
                    print(e)
                    transform_good = False
                    continue
                t_end = time.time()
                t_time = t_end - t_start
                tot_time += t_time

                if not transform_good:
                    n_fails += 1
                    continue

                if LANG == 'java':
                    # have to use a nasty hard-coded index to extract the class header
                    class_header = sample['prompt'].strip().split('\n')[6]
                    t_function = f'{class_header}\n{t_function}\n}}'
                t_program = prefix + t_function + sample["test"]

                task_id = sample['task_id'].split('/')[-1]
                mutableast_instances.append((task_id, t_program))

            print(f'Number of mutable-ast supported samples: {len(mutableast_instances)}')

            res, failures = runner.check_solutions(mutableast_instances, n_processes=16)
            n_fails += len(failures)
            n_successes += res[runner.msg_good]

            assert n_samples == n_successes + n_fails

            transform_times.append(tot_time / n_samples)
            transform_pass_rates.append(n_successes / n_samples)

            print(f'Number of samples: {n_samples}')
            print(f'Number of successes: {n_successes}')
            print(f'Number of failures: {n_fails}')
            print(f'Average time: {tot_time / n_samples*1000:.2f} ms')
            print(f'Pass rate: {n_successes / n_samples * 100:.2f}%')
            print(res, failures)
            print()

    print('Times:')
    print(transform_times)
    print('Pass rates:')
    print(transform_pass_rates)

    avg_time = sum(transform_times) / len(transform_times)
    avg_pass_rate = sum(transform_pass_rates) / len(transform_pass_rates)
    print(f'Average time: {avg_time*1000:.2f} ms')
    print(f'Average pass rate: {avg_pass_rate*100:.2f}%')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python benchmark_mbxp.py <lang>')
        exit(0)

    main(lang=sys.argv[1])
