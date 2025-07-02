INFER_DIR_LIST=("$@")

for i in "${!INFER_DIR_LIST[@]}"; do
    INFER_DIR=${INFER_DIR_LIST[$i]}
    JOB_DIR=$(dirname $INFER_DIR)
    DATASET_NAME=$(basename $(basename $INFER_DIR))
    echo "Processing $JOB_DIR with dataset $DATASET_NAME"
    python runner/eval.py \
        --root $ROOT \
        --dataset_name $DATASET_NAME \
        --job_root $JOB_DIR \
        --condition c
done
wait
