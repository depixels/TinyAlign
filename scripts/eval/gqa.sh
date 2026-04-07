#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="/data/hyz/RAT/dataset/eval/gqa"

MODEL_PATH='/data/hyz/data/TinyLLaVA-m-top10/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-frozen-vit'
MODEL_NAME='m-llavadata-tinyllava-frozen-vit'
EVAL_DIR='/data/hyz/RAT/dataset/eval'

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/gqa/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/gqa/images \
        --answers-file $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced

