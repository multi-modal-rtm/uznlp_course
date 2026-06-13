---
name: kaggle-hardware
description: Use whenever writing, sizing, or reviewing any code, model choice, training configuration, or dataset for the NLP course. All compute targets Kaggle Notebooks free tier — enforce its hardware ceilings.
---

# Kaggle Free Tier — Hard Constraints

All practice materials run on Kaggle Notebooks free tier. Verified June 2026:
single NVIDIA T4 or P100, **16 GB VRAM**, ~30 GPU-hours/week per account,
sessions up to ~9 hours with background execution. Quotas can change without
notice — never promise exact numbers in tinglovchi-facing text; say
"haftalik bepul GPU kvotangiz" and link Kaggle's docs.

## Design ceilings (binding)

- **Weeks 1–2 (days 1–8)**: design for **CPU-only** by default. Classical
  pipeline, BoW/TF-IDF, classical ML, N-grams, Word2Vec training on small
  corpora, small RNN/LSTM demos — all must complete on CPU in session time;
  GPU is an accelerator, never a requirement. This preserves quota for
  weeks 3–4 and keeps materials usable on a laptop.
- **Weeks 3–4 GPU days**: fit in 16 GB. Allowed: full fine-tuning of
  DistilBERT / BERT-base / mBERT-class models (≤ ~180M params) with
  batch sizes sized to leave ≥ 3 GB headroom; gradient accumulation instead
  of large batches. Larger models (≥ 1B): ONLY via PEFT — LoRA/QLoRA with
  4-bit quantization (bitsandbytes) — and frame this explicitly as a
  teaching point about parameter-efficient adaptation, not a workaround.
- **Never** assume A100/H100, multi-GPU, or > 16 GB anywhere, including
  "stretch" tasks.
- **RAG/agents days**: the LLM is called via a hosted API free tier, not
  hosted locally. Embedding models run locally (small sentence-transformers
  fit easily). Provide an OFFLINE_FALLBACK with cached API responses.

## Engineering rules

- Pin every package version in the env-check cell; Kaggle base images shift.
- Datasets: bundle as Kaggle Datasets attached to the notebook; per-day data
  ≤ 500 MB; document the attach step in Day 1 orientation.
- Checkpoint every training loop ≥ every 2 minutes of wall time
  (`save_pretrained` / `torch.save`) — free sessions can disconnect.
- Set all seeds (`random`, `numpy`, `torch`) in the env-check cell so
  asserts on results are stable.
- Mixed precision (`torch.cuda.amp` / fp16 in TrainingArguments) on by
  default for transformer days; explain the memory math on the slide that
  introduces it.
- Any cell expected to exceed 5 minutes must be split or downsized; give the
  full-scale configuration in a comment for tinglovchilar who rerun at home.
