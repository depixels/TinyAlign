DATA_PATH=/data/hyz/RAT/dataset/text_files/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/data/hyz/RAT/dataset/text_files/llava_v1_5_mix665k.json #finetune annotation file path
# FINETUNE_DATA_PATH=/data/hyz/RAT/dataset/text_files/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json #finetune annotation file path
IMAGE_PATH=/data/hyz/RAT/dataset/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/data/hyz/RAT/dataset #finetune image dir

LLM_VERSION=microsoft/phi-2 # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm
CN2_VERSION=remlp2x_gelu
TOP_K=10 #retrieval item nums

# bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION"  "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$CN2_VERSION" "$TOP_K"
bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION"  "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$CN2_VERSION" "$TOP_K"