# TinyAlign

TinyAlign is a retrieval-augmented lightweight vision-language modeling project built on top of [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory).

This repository contains the official code for [TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks](https://arxiv.org/abs/2505.12884), accepted to Findings of ACL 2026.

## Overview

Compared with the original TinyLLaVA codebase, this project mainly adds:

- memory bank construction for retrieval-augmented alignment
- memory retrieval during training and inference
- an extra retrieval connector (`connector2`) to inject retrieved latent features into the multimodal sequence
- `<rag>`-aware prompt/token handling in the data template layer

The goal is to improve lightweight VLM alignment by retrieving relevant latent context from an external memory bank instead of relying only on the frozen vision encoder and language model.

## Code Structure

- `tinyllava/model/modeling_tinyllava.py`: core retrieval-augmented multimodal forward path
- `tinyllava/model/configuration_tinyllava.py`: model config, including retrieval parameters
- `tinyllava/data/template/base.py`: `<image>` and `<rag>` token processing
- `tinyllava/train/train.py`: training entry
- `tinyllava/training_recipe/base.py`: checkpoint save/load, including `connector2`
- `docs/CODEBASE_OVERVIEW.md`: concise code walkthrough for this refactored release

## Installation

```bash
conda create -n tinyalign python=3.10 -y
conda activate tinyalign
pip install --upgrade pip
pip install -e .
```

If you use FlashAttention in your environment:

```bash
pip install flash-attn --no-build-isolation
```

## Retrieval Memory Bank

The retrieval path expects a memory directory containing:

- a FAISS index file, default: `Merged_faiss.index`
- a serialized value store, default: `Merged_LLaVA_Dataset_Memory.pt`

You can now configure these paths through arguments instead of editing source code:

- `--retrieval_memory_dir`
- `--retrieval_index_file`
- `--retrieval_value_file`
- `--top_rag`
- `--retrieval_text_start`
- `--retrieval_alpha`

## How Memory Is Built

This is the core addition of TinyAlign.

The memory bank is not a plain text knowledge base. Each memory item is a paired `(key, value)` built from image-caption supervision data:

- `key`: a compressed multimodal query vector
- `value`: a compact latent representation produced by a Perceiver-style encoder

In the current codebase, the reference implementation is scattered in `demo.py`, `demo1.py`, and `build_memory.ipynb`.

### Memory construction pipeline

For each image-text pair in the pretraining set:

1. Encode the image with the TinyLLaVA vision tower and the original multimodal connector.
2. Tokenize the paired caption text and obtain text token embeddings from the language model embedding table.
3. Normalize image features and text features, then fuse them with a weighted mixture:
   `multimodal_features = concat(alpha * image_features, (1 - alpha) * text_features)`
4. Compute self-similarity over the fused multimodal sequence.
5. Average the attention map and use it to compress the multimodal sequence into a single query vector.
6. Feed the original image and caption into a Perceiver multimodal encoder.
7. Use the Perceiver latent output as the retrieval value.
8. Store the pair:
   - key: compressed multimodal vector
   - value: Perceiver latent tensor

### What gets saved

After collecting all pairs:

- all keys are added into a FAISS index for nearest-neighbor search
- all values are stored in a tensor file and aligned with FAISS ids

At runtime, the model:

1. rebuilds a query vector from the current input image and text
2. retrieves top-`k` nearest memory items from FAISS
3. concatenates the retrieved values
4. projects them through `connector2`
5. injects the projected retrieval features into the multimodal token sequence

### Shape intuition from the current implementation

From your current code:

- the Perceiver value is typically a latent tensor shaped like `32 x 96`
- multiple retrieved items are concatenated along the latent width dimension
- `connector2` maps the retrieved latent features into the language model hidden size

### Source files for the original memory-building code

- `demo.py`: end-to-end prototype for constructing compressed keys and latent values
- `demo1.py`: smaller-scale memory construction example
- `build_memory.ipynb`: notebook experiments for building, saving, merging, and querying the memory bank

### Reproducible memory-building script

The repository now includes a standalone script:

`scripts/build_memory_bank.py`

It does two jobs:

1. build shard files containing `(key, value)` pairs
2. merge the shards into:
   - `Merged_faiss.index`
   - `Merged_LLaVA_Dataset_Memory.pt`

### Expected dataset format

By default, the script assumes a LLaVA-style JSON list where each sample contains:

```json
{
  "image": "relative/path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "caption or target text"}
  ]
}
```

If your caption is stored in another field, pass `--caption-field`.

### Build commands

Build shards and merge immediately:

```bash
python scripts/build_memory_bank.py \
  --model-path /path/to/tinyllava_base_checkpoint \
  --dataset-json /path/to/pretrain.json \
  --image-root /path/to/images \
  --perceiver-tokenizer /path/to/perceiver_tokenizer \
  --output-dir /path/to/memory_bank \
  --save-every 5000 \
  --merge-after-build
```

If you already built shards and only want to merge them:

```bash
python scripts/build_memory_bank.py \
  --output-dir /path/to/memory_bank \
  --merge-only
```

### Output structure

After a successful run, `output-dir` contains:

```text
memory_bank/
  shards/
    memory_shard_00000.pt
    memory_shard_00001.pt
    ...
  Merged_faiss.index
  Merged_LLaVA_Dataset_Memory.pt
```

The final merged tensor file stores:

- `keys`: normalized compressed multimodal query vectors
- `values`: Perceiver latent tensors aligned with FAISS ids

### Plug the memory bank into training or inference

Once the memory bank is built, point the model to it with:

```bash
--retrieval_memory_dir /path/to/memory_bank
```

Optionally override filenames if you changed them:

```bash
--retrieval_index_file Merged_faiss.index
--retrieval_value_file Merged_LLaVA_Dataset_Memory.pt
```

## Training

Example command structure:

```bash
python -m tinyllava.train.train \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --vision_tower google/siglip-so400m-patch14-384 \
  --connector_type mlp2x_gelu \
  --connector2_type rag2x_gelu \
  --data_path /path/to/data.json \
  --image_folder /path/to/images \
  --retrieval_memory_dir /path/to/memory_bank \
  --output_dir /path/to/output
```

If you initialize from a TinyLLaVA checkpoint and keep `connector2` separately, use:

```bash
--pretrained_model_path /path/to/base_checkpoint \
--pretrained_connector2_path /path/to/connector2
```

## Inference

The main inference entry is:

```bash
python -m tinyllava.eval.run_tiny_llava \
  --model-path /path/to/model \
  --image-file /path/to/image.jpg \
  --query "Describe the image."
```

Make sure the released model config points to the correct retrieval memory directory before inference.

## Notes on This Release

- This repository keeps the original package name `tinyllava` for code compatibility.
- The refactor removes local debug hooks and hard-coded absolute paths that were tied to the author's machine.

## Acknowledgement

This codebase is built on top of [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory). Credit goes to the original authors for the base multimodal training framework.

## Citation

```bibtex
@article{hu2025tinyalign,
  title={TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks},
  author={Hu, Yuanze and Fan, Zhaoxin and Wang, Xinyu and Li, Gen and Qiu, Ye and Yang, Zhichao and Wu, Wenjun and Wu, Kejian and Sun, Yifan and Deng, Xiaotie and Dong, Jin},
  journal={arXiv preprint arXiv:2505.12884},
  year={2025}
}
```
