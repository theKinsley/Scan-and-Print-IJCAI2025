GPU_ID="$1"
shift
INFER_DIR_LIST=("$@")

for i in "${!INFER_DIR_LIST[@]}"; do
    INFER_DIR=${INFER_DIR_LIST[$i]}
    echo "Processing $INFER_DIR on cuda:$((GPU_ID + i))"
    python runner/infer.py \
        --root $ROOT \
        --gpu $((GPU_ID + i)) \
        --save_path $INFER_DIR &
done
wait