# QA Report: Day 12 Practice (P11)

**Artifacts**: `course/practices/d12_p11_seq2seq.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m11_seq2seq_translator.py` (pedagogical)
**Bundled data**: `course/practices/d12_checkpoints/uz_en_mini.txt` (original uz-en, 40 pairs)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L11 — Seq2Seq va Attention
**Next**: L12 (Day 12 lecture — Transformer)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `json.load` OK; nbformat 4.5; 28 cells (13 code, 15 md); all `id` |
| JSON valid — solutions | **PASS** | `json.load` OK; nbformat 4.5; 28 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True, CPU)** | **PASS (local)** | SOLUTIONS exec'd in sequence on Python 3.13.14 — **10/10 asserts passed**, zero exceptions. **Both paths verified**: torch 2.10+cpu (real LSTM enc-dec + Bahdanau attention) AND forced-dictionary fallback. |
| Student stub cells compile | **PASS** | All 13 code cells `compile()` clean |
| **Locked assert (attention softmax)** | **PASS** | §4A `α = softmax([2,1,3]) = [0.245, 0.090, 0.665]` (pure-numpy, torch-independent) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L11 [I3]-slayd` (§4A) |
| Every blanked region has paired assert | **PASS** | 3 blanked cells (§4B train/translate, §4C translate+BLEU, §4D attention heatmap) → paired asserts |
| m11 contract conformance | **PASS** | `train(src,tgt,epochs,max_len)` / `translate->str` / `bleu->float` exact (contracts.py); pedagogical (no save/load, `consumed_by: []`) |
| translate / bleu structural | **PASS** | translate returns `str`; bleu returns `float ∈ [0,1]` (not exact translation / high BLEU — small data, demo-quality, honest) |
| No GPU / VRAM | **PASS** | CPU-only; small seq2seq; VRAM peak 0 GB |
| Data size | **PASS** | bundled corpus ≈ 1.5 KB (≪ 500 MB; real OPUS-100 online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads parallel corpus) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | all four artifacts verified clean |

**Overall: ALL GATES PASS**

---

## torch present locally — both paths verified

Local env has **torch 2.10.0+cpu**, so the real LSTM encoder-decoder + Bahdanau
attention path was verified end-to-end on CPU; the torchless dictionary fallback
(forced `HAS_TORCH=False`) was **also** verified.

| Path | translate("men kitob o'qidim") | BLEU (5 train) |
|---|---|---|
| torch LSTM + Bahdanau attention (30 ep) | **"i read a book"** ✓ | 0.795 |
| dictionary word-alignment (forced) | "i read i" (word-by-word, gibberish) | 0.000 |

The torch seq2seq genuinely **learns to translate** the tiny corpus
("u maktabga bordi" → "he went to school"). The dictionary fallback is intentionally
simple (honest demo-quality). The locked §4A attention softmax is pure-numpy and
path-independent.

---

## Locked / Verified Numbers

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I3]** | attention `α = softmax([2,1,3])` | **[0.245, 0.090, 0.665]** | **P11 first assert** (course_map lock) |

The §4A cell reproduces the lecture's attention softmax — **P11's first assert** —
matching course_map Day 12 paired-lecture L11 `hand_example`.

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L11 [C], timing |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, HAS_TORCH, HAS_NLTK |
| §2 Yaxlit natija | 2 | MD+Code | OK — finished translate + BLEU demo |
| §3 PRIMM periferiya | 4 | Mixed | OK — parallel corpus load + tokenize; BLEU (nltk-optional); Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked attention) + 3 blanked (4B train/translate, 4C translate+BLEU, 4D heatmap) each with paired assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m11 import/contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — SOV→SVO; attention heatmap meaning, exit ticket |

Total: 28 cells (13 code, 15 markdown). Blanked core cells: 3, each paired with an assert.

---

## torch-Optional + nltk-Optional Design

`torch`/`nltk` may be absent. m11 and the notebook branch on `HAS_TORCH`/`HAS_NLTK`:
- **Kaggle path**: LSTM encoder + Bahdanau attention (`e_i = vᵀtanh(W_q s + W_k h_i)`,
  α = softmax, context = Σα_i h_i) + LSTM decoder, teacher forcing + Adam; greedy decode.
- **Offline path** (forced/torchless): a simple **word-alignment dictionary** translator
  (full attention-seq2seq numpy BPTT is too heavy/fragile — deliberately not built).
- **bleu()**: pure-python (n-gram precision + brevity penalty); uses nltk if present,
  else the module's own BLEU.

Result: runs end-to-end with or without torch/nltk, CPU-only, no OPUS-100 download.

---

## Module Conformance (contracts.py)

```
m11 Seq2SeqTranslator (pedagogical, consumed_by: []):           provides:
  train(src_texts, tgt_texts, epochs=10, max_len=50) -> None     ✓
  translate(text: str) -> str                                     ✓  (greedy decode / dict)
  bleu(references, hypotheses) -> float                           ✓  (pure-python BLEU-4 + BP)
```
No save/load (pedagogical demo). Bahdanau attention in the torch path.

---

## Deviation from course_map.yaml

course_map Day 12 `corpus_subset: uz_en_opus100` (online, 20k, demo-quality BLEU noted)
and `gpu_required: true`. The **OFFLINE_FALLBACK** uses a small **original** uz-en parallel
corpus (`uz_en_mini.txt`, 40 pairs; OPUS-100 not bundled — license/download). Local run is
**CPU-only** (GPU is an accelerator). Demo uses 30 epochs on the tiny corpus; BLEU is
demo-quality, stated honestly. CRF/beam-search omitted (greedy sufficient pedagogically).

---

## Pending

- Full Kaggle kernel run with real OPUS-100 (20k) on GPU (torch + nltk BLEU + attention
  heatmap) — confirmed when notebooks are published as a Kaggle Dataset (Day 12, 3-iyul-2026).
- **L12** (Day 12 lecture — Transformer) is the next chronological artifact.
