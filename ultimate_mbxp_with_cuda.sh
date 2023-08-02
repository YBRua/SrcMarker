if [[ $# -lt 2 ]]; then
    echo "Usage: $0 gpu_id dataset"
    return 0
fi

echo "Using GPU $1"
echo "Using dataset $2"

LANG="unk"
if [[ $2 == "mbcpp" ]]; then
    LANG="cpp"
elif [[ $2 == "mbjp" ]]; then
    LANG="java"
elif [[ $2 == "mbjsp" ]]; then
    LANG="javascript"
else
    echo "Unknown dataset $2"
    return 1
fi

# NOTE: you may have to modify the variables below
NBITS=4
FILE=eval_mbxp.py
MODEL=transformer
VARMASK_PROB=0.5
CHECKPOINT_PATH="./ckpts/4bit_transformer_main_0_github_c_funcs/models_best.pt"

conda activate torch112

CUDA_VISIBLE_DEVICES=$1 python $FILE \
    --checkpoint_path $CHECKPOINT_PATH \
    --lang $LANG \
    --dataset $2 \
    --dataset_dir ./datasets/$2 \
    --n_bits $NBITS \
    --model_arch=$MODEL \
    --shared_encoder
