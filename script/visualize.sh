INFER_DIR=$1
COND=${2:-"uncond"}
JOB_DIR=$(dirname $INFER_DIR)
DATASET_NAME=$(basename $(basename $INFER_DIR))

python runner/visualize.py \
    --root $ROOT \
    --dataset_name $DATASET_NAME \
    --job_root $JOB_DIR \
    --condition $COND
wait