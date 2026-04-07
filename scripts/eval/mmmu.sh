#!/bin/bash
MODEL_PATH='/data/hyz/data/TinyLLaVA-m-top10/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-frozen-vit'
MODEL_NAME='m-llavadata-tinyllava-frozen-vit'
EVAL_DIR='/data/hyz/RAT/dataset/eval'

python -m tinyllava.eval.model_vqa_mmmu \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/MMMU/anns_for_eval.json \
    --image-folder $EVAL_DIR/MMMU/all_images \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python scripts/convert_answer_to_mmmu.py \
    --answers-file $EVAL_DIR/MMMU/answers/$MODEL_NAME.jsonl \
    --answers-output $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json

cd $EVAL_DIR/MMMU/eval

python main_eval_only.py --output_path $EVAL_DIR/MMMU/answers/"$MODEL_NAME"_output.json
