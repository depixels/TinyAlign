#!/bin/bash

MODEL_PATH='/data/hyz/data/TinyLLaVA-m-top10/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-frozen-vit'
MODEL_NAME='m-llavadata-tinyllava-frozen-vit'
EVAL_DIR='/data/hyz/RAT/dataset/eval'

python -m tinyllava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_result.json

