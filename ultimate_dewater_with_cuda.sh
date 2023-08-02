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
elif [[ $2 == "mbjp" ]]; then
    LANG="java"
elif [[ $2 == "csn_js" ]]; then
    LANG="javascript"
elif [[ $2 == "mbjsp" ]]; then
    LANG="javascript"
else
    echo "Unknown dataset $2"
    return 1
fi

# NOTE: you may have to modify the variables below
NBITS=4
FILE=dewatermark_gru.py
MODEL=transformer
CHECKPOINT_PATH="4bit_transformer_seed_1337_csn_js"
ATTACK_DATASET="4bit_transformer_seed_1337_csn_js"
ATTACK_CHECKPOINT="dewatermark_gru_4bit_transformer_seed_1337_csn_js"

conda activate torch112

# CUDA_VISIBLE_DEVICES=$1 python $FILE \
#     --epochs 50 \
#     --seed 42 \
#     --dataset $2 \
#     --dataset_dir ./datasets/dewatermark/$2/$CHECKPOINT_PATH \
#     --lang $LANG \
#     --model_arch=$MODEL \
#     --log_prefix $CHECKPOINT_PATH \
#     --batch_size 12 \
#     --do_train


CUDA_VISIBLE_DEVICES=$1 python $FILE \
    --seed 42 \
    --dataset $2 \
    --lang $LANG \
    --model_arch $MODEL \
    --log_prefix $CHECKPOINT_PATH \
    --batch_size 1 \
    --do_attack \
    --attack_dataset_dir ./datasets/dewatermark/$2/$ATTACK_DATASET \
    --attack_checkpoint ./ckpts/$ATTACK_CHECKPOINT/best_model.pt