GPU_ID="${1:-"0"}"
DATASET="${2:-"pku"}"

if [ "$DATASET" = "pku" ]; then
    SUM_K=96
    AUG_NUM=256
    AUG_SIM="pearson"
elif [ "$DATASET" = "cgl" ]; then
    SUM_K=48
    AUG_NUM=16
    AUG_SIM="cosine"
else
    echo "Invalid dataset: $DATASET"
    exit 1
fi

python runner/train.py \
    --root $ROOT \
    --dataset_root $DATASET_ROOT \
    --gpu $GPU_ID \
    --vis_preview True \
    --dataset $DATASET \
    --model_name "filtering" \
    --tokenizer_name "sepoint" \
    --var_order "('clses','x','y')" \
    --suppl_type "density" \
    --use_layout_encoder True \
    --pick_topk $SUM_K \
    --augment True \
    --num_augment $AUG_NUM \
    --similarity_type $AUG_SIM
