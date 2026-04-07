# TinyAlign Codebase Overview

This repository is a TinyLLaVA-based codebase extended with a retrieval-augmented alignment pipeline for lightweight vision-language models.

## What changed relative to TinyLLaVA

- A memory-bank retrieval path was added to the multimodal forward pass.
- Training and inference both inject retrieved latent features through an extra connector (`connector2`).
- The prompt template inserts a dedicated `<rag>` placeholder right after `<image>`, so retrieval features become part of the sequence construction instead of an external post-processing step.

## Main code path

- Prompt/template logic: `tinyllava/data/template/base.py`
- Model config and retrieval hyperparameters: `tinyllava/model/configuration_tinyllava.py`
- Retrieval-augmented model forward: `tinyllava/model/modeling_tinyllava.py`
- Training entry: `tinyllava/train/train.py`
- Checkpoint loading/saving for the extra connector: `tinyllava/training_recipe/base.py`
- Inference entry: `tinyllava/eval/run_tiny_llava.py`

## Retrieval pipeline in the current implementation

1. The template rewrites `<image>` into `<image><rag>`.
2. The model encodes the image with the original visual tower and connector.
3. The model slices the text tokens after `retrieval_text_start`, embeds them with the language model, and mixes text and image features with `retrieval_alpha`.
4. A compressed query vector is built from the mixed multimodal features.
5. FAISS retrieves top-`k` latent items from the memory bank.
6. Retrieved values are projected by `connector2`.
7. The projected retrieval features are inserted into the multimodal embedding sequence next to the image features.

## New configurable release-facing knobs

- `retrieval_memory_dir`
- `retrieval_index_file`
- `retrieval_value_file`
- `retrieval_text_start`
- `retrieval_alpha`
- `top_rag`
- `pretrained_connector2_path`

These were added to remove local hard-coded paths and make the repository publishable.
