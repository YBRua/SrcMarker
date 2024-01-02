# SrcMarker: Dual-Channel Source Code Watermarking via Scalable Code Transformations

> This repository provides the script to reproduce the major experiments in the paper
> *SrcMarker: Dual-Channel Source Code Watermarking via Scalable Code Transformations*

## Getting Started

- [Setting up the environment](#setting-up-the-environment)
- [Preparing datasets](#datasets)
- [Preprocessing](#preprocessing)

### Setting up the Environment

#### Installing Python Packages and Dependencies

You will (of course) need Python to execute the code

- Python 3 (Python 3.10)

The following packages are **required** to run the main training and evaluation scripts

- [PyTorch](https://pytorch.org/get-started/locally/) (PyTorch 1.12)
- [tree-sitter](https://tree-sitter.github.io/)
- [Huggingface Transformers](https://huggingface.co/)
- tqdm
- inflection
- sctokenizer

The following packages are optional, only required by certain experiment scripts

- [SrcML](https://www.srcml.org/)
  - only required if running the transform pipeline provided by RopGen

#### Building tree-sitter Parsers

We use `tree-sitter` for MutableAST construction, syntax checking and codebleu computation. Follow the steps below to build a parser for `tree-sitter`.

Notice that our current implementation of MutableAST is based on specific versions of tree-sitter parsers. The latest tree-sitter parsers might have updated their grammar, which could be incompatible with MutableAST. Therefore please checkout to the commits as is specified in the shell script below, or otherwise MutableAST might break.

```sh
# create a directory to store sources
mkdir tree-sitter
cd tree-sitter

# clone parser repositories
git clone https://github.com/tree-sitter/tree-sitter-java.git
cd tree-sitter-java
git checkout 6c8329e2da78fae78e87c3c6f5788a2b005a4afc
cd ..

git clone https://github.com/tree-sitter/tree-sitter-cpp.git
cd tree-sitter-cpp
git checkout 0e7b7a02b6074859b51c1973eb6a8275b3315b1d
cd ..

git clone https://github.com/tree-sitter/tree-sitter-javascript.git
cd tree-sitter-javascript
git checkout f772967f7b7bc7c28f845be2420a38472b16a8ee
cd ..

# go back to parent dir
cd ..

# run python script to build the parser
python build_treesitter_langs.py ./tree-sitter

# the built parser will be put under ./parser/languages.so
```

### Datasets

#### GitHub-C and GitHub-Java

The pre-processed datasets for GitHub-C and GitHub-Java (originally available [here](https://github.com/RoPGen/RoPGen)) are included in this repository, which can be found in `./datasets/github_c_funcs` and `./datasets/github_java_funcs`.

#### MBXP

The MBXP datasets are originally available at [amazon-science/mxeval](https://github.com/amazon-science/mxeval). The filtered MBXP datasets used in our project is also included in this repository.

#### CodeSearchNet

The CSN datasets are available on the project site of [CodeSearchNet](https://github.com/github/CodeSearchNet). Since CSN datasets are relatively large, they are not included here. Follow the steps below to further process the dataset after downloading it.

1. Follow the instructions on [CodeXGLUE (code summarization task)](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) to filter the dataset.
2. Run `dataset_filter.py` to filter out samples with grammar errors or unsupported features.

```sh
python dataset_filter.py java <path_to_your_csn_jsonl>
```

The results will be stored as `filename_filtered.jsonl`, rename it into `train.jsonl` or `valid.jsonl` or `test.jsonl` depending on the split and put the three files under `./datasets/csn_java` or `./datasets/csn_js`.

#### Wrapping Up

After all datasets are processed, the final directory should look like this

```txt
- datasets
    - github_c_funcs
        - train.jsonl
        - valid.jsonl
        - test.jsonl
    - github_java_funcs
        - train.jsonl
        - valid.jsonl
        - test.jsonl
    - csn_java
        - train.jsonl
        - valid.jsonl
        - test.jsonl
    - csn_js
        - train.jsonl
        - valid.jsonl
        - test.jsonl
    - mbcpp
        - test.jsonl
    - mbjp
        - test.jsonl
    - mbjsp
        - test.jsonl
```

Note that you should ensure the dataset directory is **identical** to the structure listed above, or otherwise the data-loading modules would not be able to correctly locate the datasets.

### Preprocessing

Several preprocessing steps are required before running training or evaluation scripts. These preprocessing steps provide metadata for the subsequent training/evaluation scripts.

1. Collect variable names from datasets. The script will scan through all functions in the dataset and collect their substitutable variable names (local variables and formal parameters). Results will be stored in `./datasets/variable_names_<dataset>.json`

```sh
# python collect_variable_names_jsonl.py <dataset>
python collect_variable_names_jsonl.py csn_js
python collect_variable_names_jsonl.py csn_java
python collect_variable_names_jsonl.py github_c_funcs
python collect_variable_names_jsonl.py github_java_funcs
python collect_variable_names_jsonl.py mbcpp
python collect_variable_names_jsonl.py mbjp
python collect_variable_names_jsonl.py mbjsp
```

2. Collect all feasible transforms. The script will enumerate all feasible transformation combinations for each function in the dataset. Results will be stored in `./datasets/feasible_transforms_<dataset>.json` and `tarnsforms_per_file_<dataset>.json`. Note that this process could take a while, especially for CSN-Java.

```sh
# python collect_feasible_transforms_jsonl.py <dataset>
python collect_feasible_transforms_jsonl.py csn_js
python collect_feasible_transforms_jsonl.py csn_java
python collect_feasible_transforms_jsonl.py github_c_funcs
python collect_feasible_transforms_jsonl.py github_java_funcs
python collect_feasible_transforms_jsonl.py mbcpp
python collect_feasible_transforms_jsonl.py mbjp
python collect_feasible_transforms_jsonl.py mbjsp
```

## Running the scripts

- [Training](#training)
- Evaluations
  - [Evaluation](#main-evaluation-script)
  - [MBXP evaluation](#evaluate-on-mbxp)
  - [Re-watermarking](#re-watermarking)
  - [De-watermarking](#de-watermarking)
  - [Project-level watermark verification](#project-level-verification)
- [MutableAST benchmark](#benchmarking-mutableast-on-mbxp)

### Training

`train_main.py` is responsible for all training tasks. Refer to the `parse_args()` function in `train_main.py` for more details on the arguments.

Here are some examples.

```sh
# training a 4-bit GRU model on CSN-Java
python train_main.py \
    --lang=java \
    --dataset=csn_java \
    --dataset_dir=./datasets/csn_java \
    --n_bits=4 \
    --epochs=25 \
    --log_prefix=4bit_gru_srcmarker \
    --batch_size 64 \
    --model_arch=gru \
    --shared_encoder \
    --varmask_prob 0.5 \
    --seed 1337

# training a 4-bit Transformer model on CSN-JavaScript
python train_main.py \
    --lang=javascript \
    --dataset=csn_js \
    --dataset_dir=./datasets/csn_js \
    --n_bits=4 \
    --epochs=25 \
    --log_prefix=4bit_transformer_srcmarker \
    --batch_size 64 \
    --model_arch=transformer \
    --shared_encoder \
    --varmask_prob 0.5 \
    --seed 1337 \
    --scheduler
```

Alternatively, you can also use the `script_train.sh` to conveniently start training. However, you might have to manually modify some of the arguments in it.

```sh
# by default, it trains a 4-bit GRU model on a designated dataset,
# you have to manually change some of the variables in the script to run different tasks
# such as model architectures and checkpoint names
# source script_train.sh <dataset>
source script_train.sh csn_java
```

Checkpoints will be saved in `./ckpts`.

### Evaluation

#### Main Evaluation Script

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

# --write_output controls whether to write results to ./results directory
# the results could be used in null-hypothesis test

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

Alternatively, you can also refer to `script_eval_main.sh` for more evaluation scripts. However, you may also have to manually modify some of the parameters in it.

```sh
# you have to manually change some of the variables in the script to run different tasks
# such as checkpint paths and model architectures
# source script_eval_main.sh <gpu_id> <dataset>
source script_eval_main.sh 0 csn_java
```

#### Evaluate on MBXP

Use `eval_mbxp.py` to run evaluations on MBXP. This will not only evaluate watermark accuracy, but also run execution-based tests provided by MBXP.

Note that to run MBCPP, you will need a G++ compiler available on your machine (we use gcc version 9.4.0 in our paper); to run MBJP, you will need a JDK (we use OpenJDK 8); to run MBJSP, you will need a Node runtime (we use Node v17.3.0).

```sh
python eval_mbxp.py \
    --checkpoint_path ./ckpts/path_to_model/models_best.pt \
    --lang java \
    --dataset mbjp \
    --dataset_dir ./datasets/mbjp \
    --n_bits 4 \
    --model_arch=transformer \
    --shared_encoder
```

You can also refer to the script `script_eval_mbxp.sh` for more examples, or use the script directly for running MBXP evaluations.

#### Re-watermarking

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

You can also refer to the script `script_eval_rewater.sh` for more examples, or use the script directly.

#### De-watermarking

The de-watermarking process is a bit tricky as it composes of multiple steps.

##### Step 1. Collecting data to train a de-watermarking model

Use `dewatermark_collect_data.py` to collect paired training data for the de-watermarker. This file essentially loads a trained watermarking model, and use the model to watermark the entire training/validation/test sets to acquire paired (i.e., code before and after watermarking) data to train the de-watermarker.

```sh
python dewatermark_collect_data.py \
    --dataset csn_java \
    --lang java \
    --dataset_dir ./datasets/csn_java \
    --checkpoint_path ./ckpts/path_to_shadow_model.pt \
    --model_arch transformer
```

Note that for black-box attacks, the argument `checkpoint_path` should be the path to a shadow model ($SrcMarker_{adv}$) instead of the original watermarking model, or otherwise you would be launching a white-box attack.

##### Step 2. Training the de-watermarking model

The collected data would be put in `./datasets/dewatermark/<dataset_name>/<model_checkpoint>`. We can then use the collected data to train the de-watermarking model.

```sh
# NOTE: You may want a smaller batch size because this seq2seq GRU takes up plenty of GPU memory.
python dewatermark_gru.py \
    --epochs 25 \
    --seed 42 \
    --dataset csn_java \
    --dataset_dir ./datasets/dewatermark/csn_java/shadow_model_checkpoint_name \
    --lang java \
    --log_prefix dewatermarker \
    --batch_size 12 \
    --do_train
```

The training could be rather slow due to the sequence-to-sequence modeling nature of the de-watermarker. It takes around 3 hours to train one epoch on CSN-Java on our 3090 GPU. Fortunately, the model usually converges within 3 epochs so you can early-stop it.

##### Step 3. Launching attack with the de-watermarker

After the de-watermarker is trained, you can then use it to launch an attack.

Before launching attack, first collect data from your **victim model** using `dewatermark_collect_data.py`. This will collect the watermarked code of the victim model.

```sh
python dewatermark_collect_data.py \
    --dataset csn_java \
    --lang java \
    --dataset_dir ./datasets/csn_java \
    --checkpoint_path ./ckpts/path_to_victim_model.pt \
    --model_arch transformer
```

Then use `dewatermark_gru.py` to de-watermark the outputs of the victim.

```sh
# this is used for launching attacks with the de-watermarking model
python dewatermark_gru.py \
    --seed 42 \
    --dataset csn_java \
    --lang java \
    --log_prefix dewatermarker-attack \
    --batch_size 1 \
    --do_attack \
    --attack_dataset_dir ./datasets/dewatermark/csn_java/path_to_victim_model \
    --attack_checkpoint ./ckpts/path_to_dewatermarking_model/best_model.pt
```

This will create a de-watermarked output (`.json` file) under `./ckpts/path_to_dewatermarking_model`.

##### Step 4. Evaluate on the de-watermarked output

The attack produces de-watermarked output of the watermarked code. We then evaluate the watermark extraction on the de-watermarked outputs. This is done with `dewatermark_attack.py`

```sh
python dewatermark_attack.py \
    --dataset csn_java \
    --lang java \
    --dataset_dir ./datasets/csn_java \
    --original_test_outputs ./datasets/dewatermark/csn_java/path_to_vicitim_model/test.jsonl \
    --attack_results ./ckpts/path_to_dewatermarking_model/path_to_victim_model.json \
    --model_arch transformer \
    --checkpoint_path ./ckpts/path_to_victim_model/models_best.pt
```

where

- `dataset_dir` is the path to the original dataset directory
- `original_test_outputs` is the path to the watermarked output (jsonl file produced by dewatermark_collect_data.py using the victim model)
- `attack_results` is the path to the output of the de-watermarking model (json file produced in Step 3)
- `model_arch` and `checkpoint_path` should be the architecture and path to the victim model

#### Project-level Watermark Verification

#### Project-level Watermark Aggregation

To run the project level watermark verification, you will first need to run `eval_main.py` on CSN datasets (`csn_java` or `csn_js`), with the `--write_output` argument. Please refer to the documentations in [Main Evaluation Script](#main-evaluation-script).

The evaluation script will then group the ground truths and extracted watermarks by repository and store them into a pickle file, located in `./results`. The pickle file will be named as `<checkpoint_name>_<dataset>_long.pkl`.

- For example, `./results/4bit_transformer_main_42_csn_js_long.pkl`.

If an attack is performed (e.g., 50% random variable substitution), the script will also store the watermark extracted from the attacked code in *another* pickle file, named as `<checkpint_name>_<dataset>_<attack>_long.pkl`.

- For example, `4bit_transformer_main_42_csn_js_vadv75_long.pkl`.

#### Project-level Verification

One can then verify the project-level watermark with `null_hypothesis_test.py`

```sh
# python null_hypothesis_test.py <path_to_pickle>
python null_hypothesis_test.py ./results/4bit_transformer_main_42_csn_js_long.pkl
```

### Benchmarking MutableAST on MBXP

`benchmark_mbxp.py`, `benchmark_mbxp_natgen.py` and `benchmark_mbxp_ropgen.py` are used for benchmarking mutableast, NatGen and RopGen respectively. Simply use them as

```sh
python benchmark_mbxp.py java|javascript|cpp
```

Note that you will need the corresponding compiler and/or runtime environment to run the benchmark. Further, for RopGen, you will also need to install SrcML.
