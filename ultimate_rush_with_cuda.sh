if [[ $# -lt 1 ]]; then
    echo "Usage: $0 dataset"
    return 1
fi

GPU_ID=$(python find_gpu.py)

echo "Using GPU $GPU_ID"
echo "Using dataset $1"

LANG="c"
if [[ $1 == "github_c_funcs" ]]; then
    LANG="cpp"
elif [[ $1 == "github_java_funcs" ]]; then
    LANG="java"
elif [[ $1 == "csn_java" ]]; then
    LANG="java"
else
    echo "Unknown dataset $1"
    return 1
fi

NBITS=4  # Number of bits
MODEL=gru  # Model architecture
# NOTE: also remember to change the log_prefix parameter

CUDA_VISIBLE_DEVICES=$GPU_ID python train_main.py \
    --lang=$LANG \
    --dataset=$1 \
    --dataset_dir=./datasets/$1 \
    --n_bits=$NBITS \
    --epochs=50 \
    --log_prefix="$NBITS"bit_"$MODEL"_prerelease \
    --batch_size 64 \
    --model_arch=$MODEL \
    --shared_encoder \
    --seed 42
    # --scheduler

# CUDA_VISIBLE_DEVICES=$GPU_ID python codebert_train.py \
#     --lang=$LANG \
#     --dataset=$1 \
#     --dataset_dir=./datasets/$1 \
#     --n_bits=$NBITS \
#     --epochs=15 \
#     --log_prefix=4bit_codebert \
#     --batch_size 16