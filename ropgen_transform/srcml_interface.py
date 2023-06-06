import os
import subprocess


def _exec_shell_cmd(command: str):
    subp = subprocess.Popen(command,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            encoding="utf-8")
    subp.wait(10)
    if subp.poll() != 0:
        raise RuntimeError(f'Failed to execute command: {command}')


def srcml_program_to_xml(input_code_path: str, output_xml_path: str):
    output_base_dir = os.path.dirname(output_xml_path)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    command = [
        'srcml',
        f'"{input_code_path}"',
        '-o',
        f'"{output_xml_path}"',
        '--position',
        '--src-encoding=UTF-8',
    ]

    _exec_shell_cmd(' '.join(command))


def srcml_xml_to_program(input_xml_path: str, output_code_path: str):
    output_base_dir = os.path.dirname(output_code_path)
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    command = [
        'srcml',
        f'"{input_xml_path}"',
        '-o',
        f'"{output_code_path}"',
        '--src-encoding=UTF-8',
    ]

    _exec_shell_cmd(' '.join(command))
