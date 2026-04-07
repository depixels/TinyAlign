#!/bin/bash
if [ $# -ne 12 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <CONV_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>"
    exit 1
fi

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
CONV_VERSION="$7"
VERSION="$8"
TRAIN_RECIPE="$9"
MODEL_MAX_LENGTH="${10}"
CN2_VERSION="${11}"
TOP_K="${12}"


VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"
    # --connector2_type $CN2_VERSION \
    # --top_k $TOP_K \
deepspeed --include localhost:3,4,5 --master_port 29504 tinyllava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --connector2_type $CN2_VERSION \
    --top_k $TOP_K \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower full \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --tune_type_connector2 full \
    --group_by_modality_length True \
    --pretrained_model_path /data/hyz/data/TinyLLaVA-m-top10/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-pretrain-frozen-vit-onlyCELoss \
    --output_dir /data/hyz/data/TinyLLaVA-m-top10/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune-tune-vitv2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune
