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

# NOTE: you may have to modify the variables below
NBITS=4
FILE=eval_main.py
MODEL=gru
CHECKPOINT_PATH="./ckpts/4bit_gru_tau_42_csn_java/models_best.pt"

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

# CUDA_VISIBLE_DEVICES=$1 python $FILE \
#     --checkpoint_path $CHECKPOINT_PATH \
#     --lang $LANG \
#     --dataset $2 \
#     --dataset_dir ./datasets/$2 \
#     --use_natgen \
#     --n_bits $NBITS \
#     --model_arch=$MODEL \
#     --shared_encoder

# CUDA_VISIBLE_DEVICES=$1 python $FILE \
#     --checkpoint_path $CHECKPOINT_PATH \
#     --lang $LANG \
#     --dataset $2 \
#     --dataset_dir ./datasets/$2 \
#     --n_bits $NBITS \
#     --model_arch=$MODEL \
#     --shared_encoder \
#     --all_adv

# for prop in 0.25 0.5 0.75 1.0
# do
# CUDA_VISIBLE_DEVICES=$1 python $FILE \
#     --checkpoint_path $CHECKPOINT_PATH \
#     --lang $LANG \
#     --dataset $2 \
#     --dataset_dir ./datasets/$2 \
#     --var_adv \
#     --var_adv_proportion $prop \
#     --n_bits $NBITS \
#     --model_arch=$MODEL \
#     --shared_encoder
# done

# for n_adv in 1 2 3
# do
# CUDA_VISIBLE_DEVICES=$1 python $FILE \
#     --checkpoint_path $CHECKPOINT_PATH \
#     --lang $LANG \
#     --dataset $2 \
#     --dataset_dir ./datasets/$2 \
#     --trans_adv \
#     --n_trans_adv $n_adv \
#     --n_bits $NBITS \
#     --model_arch=$MODEL \
#     --shared_encoder
# done
