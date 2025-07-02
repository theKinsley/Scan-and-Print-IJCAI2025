export CUDA_VISIBLE_DEVICES=0,1,2,3
SPLIT=("test" "valid" "train")
INFER_CSV=("test" "train" "train")
DATASET=$1
INFER_CKPT=$2

echo "Inferece on $DATASET with $INFER_CKPT"

# maps
for i in {0..1}; do
    torchrun --standalone --nnodes=1 --nproc-per-node=4 main.py \
        --dataset_root $DATASET_ROOT --dataset $DATASET \
        --infer --infer_ckpt $INFER_CKPT \
        --infer_csv "${INFER_CSV[i]}"
done