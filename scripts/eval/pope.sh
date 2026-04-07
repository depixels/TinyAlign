#!/bin/bash

MODEL_PATH='/data/hyz/data/TinyLLaVA-m-top10/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-frozen-vit'
MODEL_NAME='m-llavadata-tinyllava-frozen-vit'
EVAL_DIR='/data/hyz/RAT/dataset/eval'

python -m tinyllava.eval.model_vqa_pope \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
