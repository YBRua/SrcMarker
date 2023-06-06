if [[ $# -lt 2 ]]; then
    echo "Usage: $0 gpu_id dataset"
    return 0
fi

echo "Using GPU $1"
echo "Using dataset $2"

LANG="c"
if [[ $2 == "github_c_funcs" ]]; then
    LANG="cpp"
elif [[ $2 == "github_java_funcs" ]]; then
    LANG="java"
elif [[ $2 == "csn_java" ]]; then
    LANG="java"
else
    echo "Unknown dataset $2"
    return 1
fi

NBITS=4
FILE=eval_natgen.py
MODEL=transformer
CHECKPOINT_PATH="./ckpts/4bit_transformer_tau_42_csn_java/models_best.pt"

conda activate torch112

CUDA_VISIBLE_DEVICES=$1 python $FILE \
    --checkpoint_path $CHECKPOINT_PATH \
    --lang $LANG \
    --dataset $2 \
    --dataset_dir ./datasets/$2 \
    --n_bits $NBITS \
    --model_arch=$MODEL \
    --shared_encoder \
    --write_output
