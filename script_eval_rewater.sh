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
elif [[ $2 == "csn_js" ]]; then
    LANG="javascript"
elif [[ $2 == "mbjp" ]]; then
    LANG="java"
elif [[ $2 == "mbjsp" ]]; then
    LANG="javascript"
else
    echo "Unknown dataset $2"
    return 1
fi

FILE=eval_rewater.py
CHECKPOINT_PATH="./ckpts/4bit_transformer_seed_1337_csn_js/models_best.pt"
ADV_PATH="./ckpts/4bit_transformer_seed_1337_csn_js/models_best.pt"
conda activate torch112
CUDA_VISIBLE_DEVICES=$1 python $FILE \
    --checkpoint_path $CHECKPOINT_PATH \
    --adv_path $ADV_PATH \
    --lang $LANG \
    --dataset $2 \
    --dataset_dir ./datasets/$2 \
    --n_bits 4 \
    --model_arch transformer \
    --shared_encoder
