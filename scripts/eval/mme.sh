#!/bin/bash

MODEL_PATH='/data/hyz/data/TinyLLaVA-m-top10/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-frozen-vit'
MODEL_NAME='m-llavadata-tinyllava-frozen-vit'
EVAL_DIR='/data/hyz/RAT/dataset/eval'

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MME/llava_mme.jsonl \
    --image-folder $EVAL_DIR/MME/MME_Benchmark_release_version \
    --answers-file $EVAL_DIR/MME/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
   --conv-mode llama

cd $EVAL_DIR/MME

python convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

python calculation.py --results_dir answers/$MODEL_NAME

