# TinyAlign

TinyAlign is a retrieval-augmented lightweight vision-language modeling project built on top of [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory).

This repository implements the codebase behind the paper [TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks](https://arxiv.org/abs/2505.12884). According to the project status provided by the author, this work was accepted to Findings of ACL 2026.

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
- `gcc-10.2.0.tar.gz` is not part of the actual TinyAlign method and should not be included in a clean public release.

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
