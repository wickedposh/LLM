# Scaling Gordon

A research-engineering extension of the Gordon Ramsay nanoGPT learning project.

## Milestone 1: reproducible single-device training

This directory provides a configurable decoder-only Transformer with causal scaled-dot-product attention, byte-level input encoding, AdamW training, checkpointing, and JSON throughput metrics. It does **not** yet implement distributed training.

### Install

From the repository root, create a Python environment and install a current PyTorch release using the [official selector](https://pytorch.org/get-started/locally/). On Apple Silicon, verify:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Train on the M5 GPU

Provide a local UTF-8 text corpus. For the original Gordon Ramsay dataset, point `--data` at your local `ramsey.txt` file (not committed here).

```bash
python -m scaling_gordon.train --data /path/to/ramsey.txt --device mps --steps 200 --output runs/mps
python -m scaling_gordon.train --data /path/to/ramsey.txt --device cpu --steps 200 --output runs/cpu
```

Compare `runs/mps/metrics.json` with `runs/cpu/metrics.json`. Each run writes a checkpoint. For small models CPU can be faster because device overhead dominates. GPU timing is synchronized. CPU and GPU timings include setup-independent training loop and batch transfer, but do not include startup or model initialization.

This baseline uses raw UTF-8 bytes (vocabulary 256), **not** the original learned BPE merges. Results are therefore not directly comparable to the old tokenizer's losses. The original implementation remains on the `chatbot` branch.

### Planned milestones

1. GPU memory profiling and model-size sweeps.
2. Gradient accumulation, mixed precision where supported, and activation checkpointing.
3. Reproducible experiments and tests.
4. CUDA profiling on NVIDIA hardware.
5. DDP and FSDP with multi-GPU benchmarking.

### Known limitations

- Validation is a single sampled batch, not a full evaluation.
- No resume-from-checkpoint yet.
- No learning-rate schedule or gradient clipping.
- MPS metric reports **current** allocated memory, not peak usage.
- Performance and correctness on the user's M5 have not yet been tested.
