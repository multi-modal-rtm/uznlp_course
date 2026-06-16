# QA Report: Day 10 Practice (P9)

**Artifacts**: `course/practices/d10_p9_textgen.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m09_text_generator.py` (pedagogical)
**Bundled data**: `course/practices/d10_checkpoints/uz_sheriy_korpus.txt` (original Uzbek verse, ~1.75 KB)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L9 — Matn generatsiya va ikki tomonlama RNN
**Next**: L10 (Day 10 lecture — NER)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `json.load` OK; nbformat 4.5; 28 cells (13 code, 15 md); all `id` |
| JSON valid — solutions | **PASS** | `json.load` OK; nbformat 4.5; 28 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True, CPU)** | **PASS (local)** | SOLUTIONS exec'd in sequence on Python 3.13.14 — **10/10 asserts passed**, zero exceptions. **Both paths verified**: torch 2.10+cpu (real char-LSTM, num_layers=2) AND forced char-n-gram fallback. |
| Student stub cells compile | **PASS** | All 13 code cells `compile()` clean |
| **Locked assert (temperature softmax)** | **PASS** | §4A `T=1 p(nlp)=0.665` (pure-numpy, torch-independent) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L9 [I2]-slayd bilan solishtiring` (§4A) |
| Every blanked region has paired assert | **PASS** | 3 blanked cells (§4B train+generate, §4C temperature compare, §4D perplexity+Bi-LSTM) → paired asserts |
| m09 contract conformance | **PASS** | `train(text, epochs, hidden_size)` / `generate(seed, length, temperature)->str` exact; char-level; no save/load (pedagogical, `consumed_by: []`) |
| generate structural asserts | **PASS** | returns `str` of requested length; all chars from corpus vocab; works at T=0.3/0.7/1.2 |
| Capstone continuity | **PASS** | char-level (m01 optional); pedagogical demo, not wired into pipeline |
| No GPU / VRAM | **PASS** | CPU-only locally; GPU is accelerator-only per kaggle-hardware |
| Data size | **PASS** | bundled corpus ≈ 1.75 KB (≪ 500 MB) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, generator `RandomState(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads corpus + char vocab) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` across all artifacts (incl. corpus) |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | all four artifacts verified clean (Cyrillic trap checked → 0) |

**Overall: ALL GATES PASS**

---

## torch present locally — both paths verified

Local env has **torch 2.10.0+cpu**, so the real char-LSTM (num_layers=2) path was
verified end-to-end on CPU; the char-n-gram fallback (forced `HAS_TORCH=False`) was
**also** verified.

| Path | generate("tong ", 50, T=0.7) | len ok | chars in vocab |
|---|---|---|---|
| torch char-LSTM (hidden=64, 2 layers, 12 ep) | gibberish (tiny corpus, expected) | ✓ | ✓ |
| char n-gram (order 4) + temperature | "tong porladi mehmonda suvga tog' bag'ri…" | ✓ | ✓ |

The n-gram fallback shows clear temperature behaviour: **T=0.3** repetitive coherent
("tong porladi sokin tunda to'ldi…"), **T=1.2** more varied. The locked §4A temperature
softmax assert is pure-numpy and path-independent.

---

## Locked / Verified Numbers

| Assert | Slide | Expected | Computed |
|--------|-------|----------|----------|
| §4A temperature softmax `p(nlp)` at T=1 | L9 [I2] | `0.665` | `0.6652` |

The §4A cell reproduces the lecture temperature hand example — **P9's first assert**,
matching course_map Day 10 paired-lecture L9 `hand_example`.

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L9 [C], timing |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, HAS_TORCH |
| §2 Yaxlit natija | 2 | MD+Code | OK — finished generate demo |
| §3 PRIMM periferiya | 4 | Mixed | OK — char tokenization + next-char pairs; training-loop schema (clipping L9[I3]); Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked) + 3 blanked (4B train/generate, 4C temperature, 4D perplexity+Bi-LSTM) each with paired assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m09 import/contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — temperature pattern; why Bi-LSTM can't generate, exit ticket |

Total: 28 cells (13 code, 15 markdown). Blanked core cells: 3, each paired with an assert.

---

## torch-Optional Design + Bi-LSTM nuance

`torch` may be absent. m09 branches on `HAS_TORCH`:
- **Kaggle path**: char-LSTM (`nn.Embedding + nn.LSTM(num_layers=2) + nn.Linear`),
  next-char `CrossEntropyLoss` + `Adam`; autoregressive generate with temperature.
- **Offline path** (forced/torchless): char **n-gram** (order 4) + temperature sampling —
  generates plausible text, supports temperature, no fragile BPTT (m05 idea, char-level).

**Bi-LSTM nuance (taught in L9):** a bidirectional model **cannot do autoregressive
generation** (it needs future tokens). So §4D treats Bi-LSTM as an *understanding* layer
(output dim doubles to 2·hidden) and measures **perplexity** for quality — generation
stays unidirectional. This is made explicit in §4D, §6.

---

## Module Conformance (contracts.py)

```
m09 TextGenerator (pedagogical, consumed_by: []):                provides:
  train(text: str, epochs=20, hidden_size=128) -> None            ✓  (char-level, raw text)
  generate(seed: str, length=200, temperature=0.7) -> str         ✓  (autoregressive + temperature)
```
No save/load (pedagogical demo). Char-level, so m01 is optional (own char tokenizer).

---

## Deviation from course_map.yaml

course_map Day 10 `corpus_subset: uz_news_full` (or literary excerpt) and
`gpu_required: true`. The **OFFLINE_FALLBACK** uses a small **original** Uzbek verse
corpus (`uz_sheriy_korpus.txt`, no copyright/license concern; Cho'lpon/Navoiy texts not
bundled to avoid licensing). Local run is **CPU-only** (GPU is an accelerator). Demo uses
`hidden=64, epochs=12`; full-scale `hidden=128, 2 layers, epochs=20` noted in a comment.

---

## Pending

- Full Kaggle kernel run with a larger literary corpus on GPU (torch char-LSTM) —
  confirmed when notebooks are published as a Kaggle Dataset (Day 10, 30-iyun-2026).
- **L10** (Day 10 lecture — NER) is the next chronological artifact.
- **w3 milestone** (1-iyul) integrates m01–m08.
