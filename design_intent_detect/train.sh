export CUDA_VISIBLE_DEVICES=6,7,8,9
DATASET=$1
if [ "$DATASET" = "pku" ]; then
    LR=1e-4
    BATCH_SIZE=64
    EPOCH=71
    ADDITIONAL_ARGS="--use_con_loss"
elif [ "$DATASET" = "cgl" ]; then
    LR=1e-4
    BATCH_SIZE=128
    EPOCH=26
    ADDITIONAL_ARGS="--use_con_loss"
elif [ "$DATASET" = "all" ]; then
    LR=1e-6
    BATCH_SIZE=128
    EPOCH=26
    ADDITIONAL_ARGS=""
fi

echo "Training on $DATASET with $EPOCH epochs"

torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py \
        --dataset_root $DATASET_ROOT --dataset $1 \
        --batch_size $BATCH_SIZE --learning_rate $LR --model_dm_act "none" \
        --epoch $EPOCH \
        ${ADDITIONAL_ARGS}