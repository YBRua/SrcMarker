# SrcMarker: Dual-Channel Source Code Watermarking via Scalable Code Transformations

> This repository provides the script to reproduce the major experiments in the paper
> *SrcMarker: Dual-Channel Source Code Watermarking via Scalable Code Transformations*

## Getting Started

### Getting the Code

It seems this anonymous repository does not support downloading all files simultaneously. Therefore we have packed all source code in this repository into `SrcMarker.zip`. The zip file contains exactly the same files as in this repository, so it would be a bit more convenient if you want to download all source code.

### Setting up the Environment

#### Installing Python Packages and Dependencies

You will (of course) need Python to execute the code

- Python 3 (Python 3.10)

The following packages are **required** to run the main training and evaluation scripts

- [PyTorch](https://pytorch.org/get-started/locally/) (We have used PyTorch 1.12 in our experiments)
- [tree-sitter](https://tree-sitter.github.io/)
- [Huggingface Transformers](https://huggingface.co/)
- tqdm
- inflection
- sctokenizer

Use pip or conda to install all the packages above.

The followings are optional, only required by certain experiment scripts

- [SrcML](https://www.srcml.org/)
  - only required if running the transform pipeline provided by RopGen

Note that SrcML is NOT a Python package, it is instead a commandline interface and should be directly installed on your machine.

#### Building tree-sitter Parsers

We use `tree-sitter` for MutableAST construction, syntax checking and codebleu computation. Follow the steps below to build a parser for `tree-sitter`.

```sh
# create a directory to store sources
mkdir tree-sitter
cd tree-sitter

# clone parser repositories
git clone https://github.com/tree-sitter/tree-sitter-java.git
git clone https://github.com/tree-sitter/tree-sitter-cpp.git
git clone https://github.com/tree-sitter/tree-sitter-c.git

# go back to parent dir
cd ..

# run python script to build the parser
python build_treesitter_langs.py ./tree-sitter

# the built parser will be put under ./parser/languages.so
```

### Datasets

#### GitHub-C and GitHub-Java

The processed datasets for GitHub-C and GitHub-Java (originally available [here](https://github.com/RoPGen/RoPGen)) are included in this repository, which can be found in `./datasets/github_c_funcs` and `./datasets/github_java_funcs`.

#### CodeSearchNet

The CSN-Java dataset is available on the project site of [CodeSearchNet](https://github.com/github/CodeSearchNet). Follow the steps below to further process the dataset after downloading it.

1. Follow the instructions on [CodeXGLUE (code summarization task)](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) to filter the dataset.
2. Run `dataset_filter.py` to filter out samples with grammar errors or unsupported features.

```sh
python dataset_filter.py java <path_to_your_csn_java_jsonl>
```

The results will be stored as `filename_filtered.jsonl`, rename it into `train.jsonl` or `valid.jsonl` or `test.jsonl` depending on the split and put the three files under `./datasets/csn_java`.

### Running the scripts

#### Preprocessing

Several preprocessing steps are required before running training or evaluation scripts. These preprocessing steps provide metadata for the subsequent training/evaluation scripts.

1. Collect variable names from datasets. The script will scan through all functions in the dataset and collect their substitutable variable names (local variables and formal parameters). Results will be stored in `./datasets/variable_names_<dataset>.json`

```sh
# python collect_variable_names_jsonl.py <dataset>
python collect_variable_names_jsonl.py csn_java
python collect_variable_names_jsonl.py github_c_funcs
python collect_variable_names_jsonl.py github_java_funcs
```

2. Collect all feasible transforms. The script will enumerate all feasible transformation combinations for each function in the dataset. Results will be stored in `./datasets/feasible_transforms_<dataset>.json` and `tarnsforms_per_file_<dataset>.json`. Note that this process could take a while, especially for CSN-Java.

```sh
# python collect_feasible_transforms_jsonl.py <dataset>
python collect_feasible_transforms_jsonl.py csn_java
python collect_feasible_transforms_jsonl.py github_c_funcs
python collect_feasible_transforms_jsonl.py github_java_funcs
```

#### Training

`train_main.py` is responsible for all training tasks. Refer to the `parse_args()` function in `train_main.py` for more details on the arguments.

Here are some examples.

```sh
# training a 4-bit GRU model on CSN-Java
python train_main.py \
    --lang=java \
    --dataset=csn_java \
    --dataset_dir=./datasets/csn_java \
    --n_bits=4 \
    --epochs=50 \
    --log_prefix=4bit_gru \
    --batch_size 64 \
    --model_arch=gru \
    --shared_encoder \
    --seed 42

# training a 4-bit Transformer model on CSN-Java
python train_main.py \
    --lang=java \
    --dataset=csn_java \
    --dataset_dir=./datasets/csn_java \
    --n_bits=4 \
    --epochs=50 \
    --log_prefix=4bit_transformer \
    --batch_size 64 \
    --model_arch=transformer \
    --shared_encoder \
    --seed 42 \
    --scheduler
```

Alternatively, you can also use the `ultimate_rush_with_cuda.sh` to conveniently start training. However, you may have to manually modify some of the parameters in it.

```sh
# by default, it trains a 4-bit GRU model on a designated dataset,
# you have to manually change some of the variables in the script to run different tasks
# . ultimate_rush_with_cuda.sh <dataset>
source ultimate_rush_with_cuda.sh csn_java
```

Checkpoints will be saved in `./ckpts`.

#### Evaluation

##### Main Evaluation Script

`eval_main.py` is reponsible for most of the evaluation tasks.

```sh
# run an evaluation on CSN-Java, using some 4-bit GRU checkpoints
python eval_main.py \
    --checkpoint_path <path_to_model_checkpoint> \
    --lang java \
    --dataset csn_java \
    --dataset_dir ./datasets/csn_java \
    --n_bits 4 \
    --model_arch=gru \
    --shared_encoder \
    --write_output

# run random variable substitution attack with 25% proportion
python eval_main.py \
    --checkpoint_path <path_to_model_checkpoint> \
    --lang java \
    --dataset csn_java \
    --dataset_dir ./datasets/csn_java \
    --n_bits 4 \
    --model_arch=gru \
    --shared_encoder \
    --var_adv \
    --var_adv_proportion 0.25 \

# run random code transformation attack with at most 2 transforms
python eval_main.py \
    --checkpoint_path <path_to_model_checkpoint> \
    --lang java \
    --dataset csn_java \
    --dataset_dir ./datasets/csn_java \
    --n_bits 4 \
    --model_arch=gru \
    --shared_encoder \
    --trans_adv \
    --n_trans_adv 2
```

Alternatively, you can also refer to `ultimate_eval_with_cuda.sh` for more evaluation scripts. However, you may also have to manually modify some of the parameters in it.

```sh
# you have to manually change some of the variables in the script to run different tasks
# . ultimate_eval_with_cuda.sh <gpu_id> <dataset>
source ultimate_eval_with_cuda.sh 0 csn_java
```

##### NatGen and ONION

Evaluation with NatGen and ONION are available in `eval_natgen.py` and `eval_onion.py`. The two scripts can be executed in a similar way.

```sh
python eval_natgen.py or eval_onion.py \
    --checkpoint_path <path_to_model_checkpoint> \
    --lang java \
    --dataset csn_java \
    --dataset_dir ./datasets/csn_java \
    --n_bits 4 \
    --model_arch=gru \
    --shared_encoder \
    --write_output
```

You can also use `ultimate_natgen_with_cuda.sh` or `ultimate_onion_with_cuda.sh` for one-step evaluation. However, you may also have to manually modify some of the parameters in it.

Also note that NatGen and ONION both requires huggingface `transformers` to run, and they can be extremely slow (~2-5 hrs for CSN-Java)

##### Rewatermarking

Evaluation for re-watermarking attack is available in `eval_rewater.py`.

```sh
python eval_rewater.py \
    --checkpoint_path <path_to_model_checkpoint> \
    --adv_path <path_to_adversarial_checkpoint> \
    --lang java \
    --dataset csn_java \
    --dataset_dir ./datasets/csn_java \
    --n_bits 4 \
    --shared_encoder
```
