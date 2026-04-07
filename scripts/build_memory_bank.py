import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import PerceiverConfig, PerceiverTokenizer
from transformers.models.perceiver.modeling_perceiver import PerceiverEmbeddings
from transformers.models.perceiver.modeling_perceiver import PerceiverEncoder
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor
from transformers.models.perceiver.modeling_perceiver import PerceiverMultimodalPreprocessor
from transformers.models.perceiver.modeling_perceiver import PerceiverTextPreprocessor

from tinyllava.data.image_preprocess import ImagePreprocess
from tinyllava.model import TinyLlava, TinyLlavaConfig
from tinyllava.model.load_model import load_base_ckp_for_lora
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX


class PerceiverMultiModalEncoder(nn.Module):
    def __init__(self, config: PerceiverConfig):
        super().__init__()
        image_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="conv1x1",
            spatial_downsample=1,
            out_channels=256,
            position_encoding_type="trainable",
            concat_or_add_pos="concat",
            project_pos_dim=256,
            trainable_position_encoding_kwargs=dict(
                num_channels=256,
                index_dims=config.image_size ** 2,
            ),
        )
        text_preprocessor = PerceiverTextPreprocessor(config)
        self.preprocessor = PerceiverMultimodalPreprocessor(
            modalities={"image": image_preprocessor, "text": text_preprocessor},
            min_padding_size=4,
        )
        self.embeddings = PerceiverEmbeddings(config)
        self.encoder = PerceiverEncoder(config, kv_dim=self.preprocessor.num_channels)

    def forward(self, inputs):
        processed_inputs, _, _ = self.preprocessor(inputs)
        latent = self.embeddings(processed_inputs.shape[0])
        encoder_outputs = self.encoder(latent, inputs=processed_inputs)
        return encoder_outputs[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Build TinyAlign retrieval memory bank.")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--dataset-json", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--perceiver-tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--caption-prefix", type=str, default=" the caption of the image is: ")
    parser.add_argument("--image-field", type=str, default="image")
    parser.add_argument("--caption-field", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--merge-after-build", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--num-latents", type=int, default=32)
    parser.add_argument("--d-latents", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-self-attends-per-block", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--num-self-attention-heads", type=int, default=8)
    parser.add_argument("--num-cross-attention-heads", type=int, default=8)
    parser.add_argument("--qk-channels", type=int, default=96)
    parser.add_argument("--v-channels", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--max-position-embeddings", type=int, default=512)
    return parser.parse_args()


def get_torch_dtype(dtype_name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(items, sep):
        return [ele for sublist in zip(items, [sep] * len(items)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if prompt_chunks and prompt_chunks[0] and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for chunk in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(chunk[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def extract_caption(sample, caption_field=None):
    if caption_field and caption_field in sample:
        return sample[caption_field]

    conversations = sample.get("conversations")
    if not conversations:
        raise KeyError("Cannot infer caption: sample has no `conversations` and no `caption_field` was provided.")

    for message in conversations:
        if message.get("from") in {"gpt", "assistant"}:
            return message["value"]
    raise KeyError("Cannot infer caption from `conversations`.")


def load_tinyllava_base_model(model_path, device, dtype):
    config = TinyLlavaConfig.from_pretrained(model_path)
    model = TinyLlava(config)

    language_model_ckp = load_base_ckp_for_lora(os.path.join(model_path, "language_model", "pytorch_model.bin"))
    vision_tower_ckp = load_base_ckp_for_lora(os.path.join(model_path, "vision_tower", "pytorch_model.bin"))
    connector_ckp = load_base_ckp_for_lora(os.path.join(model_path, "connector", "pytorch_model.bin"))

    model.language_model.load_state_dict(language_model_ckp)
    model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
    model.connector.load_state_dict(connector_ckp, strict=False)

    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def build_perceiver(args, device):
    config = PerceiverConfig(
        num_latents=args.num_latents,
        d_latents=args.d_latents,
        d_model=args.d_model,
        num_self_attends_per_block=args.num_self_attends_per_block,
        num_blocks=args.num_blocks,
        num_self_attention_heads=args.num_self_attention_heads,
        num_cross_attention_heads=args.num_cross_attention_heads,
        qk_channels=args.qk_channels,
        v_channels=args.v_channels,
        image_size=args.image_size,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
    )
    model = PerceiverMultiModalEncoder(config).to(device)
    model.eval()
    return model


def save_shard(shard_dir, shard_id, keys, values):
    shard_path = shard_dir / f"memory_shard_{shard_id:05d}.pt"
    torch.save({"keys": torch.cat(keys, dim=0), "values": torch.cat(values, dim=0)}, shard_path)
    return shard_path


def build_memory_bank(args):
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = get_torch_dtype(args.dtype)

    with open(args.dataset_json, "r") as f:
        dataset = json.load(f)

    if args.end is None:
        args.end = len(dataset)
    if args.max_samples is not None:
        args.end = min(args.end, args.start + args.max_samples)

    model = load_tinyllava_base_model(args.model_path, device, dtype)
    tokenizer = model.tokenizer
    image_preprocess = ImagePreprocess(model.vision_tower._image_processor, model.config)
    perceiver_tokenizer = PerceiverTokenizer.from_pretrained(args.perceiver_tokenizer)
    perceiver_model = build_perceiver(args, device)

    shard_keys = []
    shard_values = []
    shard_id = 0

    iterable = range(args.start, args.end)
    for idx in tqdm(iterable, desc="Building memory"):
        sample = dataset[idx]
        image_path = Path(args.image_root) / sample[args.image_field]
        caption = extract_caption(sample, args.caption_field)

        with torch.no_grad():
            image = Image.open(image_path).convert("RGB")
            image_tensor = image_preprocess(image).unsqueeze(0).to(device=device, dtype=dtype)

            caption_prompt = "<image>\n" + args.caption_prefix + caption
            img_cap_token = tokenizer_image_token(
                caption_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).to(device)

            image_features = model.encode_images(image_tensor).squeeze(0)
            text_embedding = model.language_model.model.embed_tokens(img_cap_token[1:])

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            multimodal_features = torch.cat(
                (args.alpha * image_features, (1 - args.alpha) * text_embedding),
                dim=0,
            )

            attention_weights = torch.softmax(
                torch.matmul(multimodal_features, multimodal_features.transpose(0, 1)),
                dim=-1,
            )
            compressed_vector = torch.matmul(attention_weights.mean(dim=0), multimodal_features).unsqueeze(0)

            text_ids = perceiver_tokenizer(caption, return_tensors="pt")["input_ids"].to(device)
            latent = perceiver_model({"image": image_tensor, "text": text_ids})

            shard_keys.append(compressed_vector.to("cpu", dtype=torch.float32))
            shard_values.append(latent.to("cpu", dtype=torch.float32))

        if len(shard_keys) >= args.save_every:
            save_shard(shard_dir, shard_id, shard_keys, shard_values)
            shard_keys, shard_values = [], []
            shard_id += 1

    if shard_keys:
        save_shard(shard_dir, shard_id, shard_keys, shard_values)


def merge_memory_shards(output_dir):
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_files = sorted(shard_dir.glob("memory_shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    values = []
    keys = []
    index = None

    for shard_file in tqdm(shard_files, desc="Merging shards"):
        shard = torch.load(shard_file, map_location="cpu")
        shard_keys = shard["keys"].to(torch.float32)
        shard_values = shard["values"].to(torch.float32)

        key_array = shard_keys.numpy()
        faiss.normalize_L2(key_array)
        if index is None:
            index = faiss.IndexFlatIP(key_array.shape[1])
        index.add(key_array)

        keys.append(torch.from_numpy(key_array.copy()))
        values.append(shard_values)

    merged_keys = torch.cat(keys, dim=0)
    merged_values = torch.cat(values, dim=0)

    faiss.write_index(index, str(output_dir / "Merged_faiss.index"))
    torch.save(
        {"keys": merged_keys, "values": merged_values},
        output_dir / "Merged_LLaVA_Dataset_Memory.pt",
    )


def main():
    args = parse_args()

    if not args.merge_only:
        required = {
            "--model-path": args.model_path,
            "--dataset-json": args.dataset_json,
            "--image-root": args.image_root,
            "--perceiver-tokenizer": args.perceiver_tokenizer,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing required arguments for build mode: {', '.join(missing)}")
        build_memory_bank(args)

    if args.merge_only or args.merge_after_build:
        merge_memory_shards(args.output_dir)


if __name__ == "__main__":
    main()
