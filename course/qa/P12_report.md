# QA Report: Day 13 Practice (P12)

**Artifacts**: `course/practices/d13_p12_transformer.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m12_transformer_summarizer.py` (REAL module — `consumed_by: [15, 16]`)
**Bundled data**: `course/practices/d13_checkpoints/uz_wiki_summ_mini.txt` (original uz maqola-xulosa, 16 pairs)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L12 — Transformer arxitekturasi va matnni umumlashtirish
**Next**: L13 (Day 13 lecture — Transfer Learning, BERT/T5)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `nbformat.validate` OK; nbformat 4.5; 31 cells (15 code, 16 md); all `id` |
| JSON valid — solutions | **PASS** | `nbformat.validate` OK; nbformat 4.5; 31 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True, CPU)** | **PASS (local)** | SOLUTIONS exec'd in sequence on Python 3.13.14 — **13/13 asserts passed**, zero exceptions. **Both paths verified**: torch 2.10+cpu (real `nn.Transformer` enc-dec + sinusoidal PE) AND forced-extractive fallback (incl. save/load). |
| Student stub cells compile | **PASS** | All 15 code cells `compile()` clean |
| **Locked assert (ROUGE-1)** | **PASS** | §4A `rouge1(["nlp juda qiziq va foydali"], ["nlp juda foydali"])` → `P=1.000, R=0.600, F1=0.750` (pure-python, torch-independent) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L12 [I4]-slayd` (§4A); also `[I1]` (§4B attention), `[I2]` (§4C PE) |
| Every blanked region has paired assert | **PASS** | 3 blanked core cells (§4B scaled_dot_product_attention, §4C positional_encoding, §4D multi_head_attention) → 3 paired asserts; §4E mustaqil → structural assert |
| m12 contract conformance | **PASS** | `train(src,tgt,epochs,d_model,nhead)` / `summarize(text,max_length)->str` / `rouge1(refs,hyps)->dict` / `save` / `load` exact (contracts.py) |
| **rouge1 returns dict (not float)** | **PASS** | `{"precision","recall","f1"}`, each `float ∈ [0,1]` — distinct from m11's `bleu()->float` |
| **save/load test (real module)** | **PASS** | §5 `save → load → summarize` verified on BOTH torch (pickle state_dict) and extractive (pickle df) paths |
| summarize / rouge1 structural | **PASS** | summarize returns `str`; rouge1 returns 3-key dict in [0,1] (not exact summary / high ROUGE — small data, demo-quality, honest) |
| PE / attention numpy demo | **PASS** | `scaled_dot_product_attention` α=[0.245,0.090,0.665] (L12 [I1]); `positional_encoding(0)`=[0,1,0,1] (L12 [I2]); multi-head shape (n,d) preserved |
| No GPU / VRAM | **PASS** | CPU-only; small Transformer (d_model=64 demo); VRAM peak 0 GB |
| Data size | **PASS** | bundled corpus ≈ 1.6 KB (≪ 500 MB; real Wikipedia uz online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads parallel corpus) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` (all 4 artifacts) |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | all artifacts verified clean (notebooks, module, corpus) |

**Overall: ALL GATES PASS**

---

## torch present locally — both paths verified

Local env has **torch 2.10.0+cpu**, so the real `nn.Transformer` encoder-decoder + sinusoidal
positional encoding path was verified end-to-end on CPU (causal bool mask + key-padding masks, teacher
forcing, greedy decode); the torchless **extractive** fallback (forced `HAS_TORCH=False`) was **also**
verified, including `save`/`load`.

| Path | summarize(maqolalar[0]) | train time |
|---|---|---|
| torch `nn.Transformer` (20 ep, d_model=64) | **"Toshkent O'zbekiston poytaxti va eng yirik shahri."** ✓ | ~1–2 s (16 pairs, CPU) |
| extractive (forced, frequency-ranked) | "Toshkent O'zbekistonning poytaxti hisoblanadi. ..." (sentence selection) | instant |

The torch Transformer genuinely learns to summarize the tiny corpus. The extractive fallback is
intentionally simple (honest demo-quality). The locked §4A ROUGE-1 is pure-python and path-independent.

---

## Locked / Verified Numbers

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I4]** | ROUGE-1 (ref 5 tok, hyp 3 tok, overlap 3) | **P=1.000, R=0.600, F1=0.750** | **P12 first assert** (course_map lock) |
| [I1] | self-attn `α = softmax([2,1,3])` (via scaled_dot_product) | [0.245, 0.090, 0.665] | §4B assert |
| [I2] | `positional_encoding(pos=0, d=4)` | [0, 1, 0, 1] | §4C assert |

§4A reproduces the lecture's ROUGE-1 — **P12's first assert** — matching course_map Day 13 paired-lecture
L12 `hand_example`. §4B/§4C additionally tie the taught numpy attention/PE to L12 [I1]/[I2].

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L12 [C], timing, CC-BY-SA license disclosure |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, HAS_TORCH, HAS_SP, module path, versions |
| §2 Yaxlit natija | 2 | MD+Code | OK — load_pairs + train + summarize + ROUGE demo |
| §3 PRIMM periferiya | 3 | Mixed | OK — tokenizer (sentencepiece-optional) + scaffold stats; Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 12 | Mixed | OK — Namuna (4A locked ROUGE) + 3 blanked (4B attn, 4C PE, 4D multi-head) each + assert; 4E mustaqil + structural assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m12 contract test, save/load test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — extractive vs abstractive; Uzbek agglutination; exit ticket |

Total: 31 cells (15 code, 16 markdown). Blanked core cells: 3, each paired with an assert.

---

## torch-Optional + sentencepiece-Optional Design

`torch`/`sentencepiece` may be absent. m12 and the notebook branch on `HAS_TORCH`/`HAS_SP`:
- **Kaggle path**: mini `nn.Transformer` (d_model=128, nhead=4, 2+2 layers) + sinusoidal PE + teacher
  forcing + Adam; greedy decode; BPE tokenizer (sentencepiece, vocab=5000).
- **Offline path** (forced/torchless): a frequency-ranked **extractive** summarizer (full Transformer
  numpy BPTT deliberately NOT built — too heavy/fragile). Word-level tokenizer when sentencepiece absent.
- **rouge1()**: pure-python (unigram Counter clipping → per-pair P/R/F1 → corpus average), returns dict.

Result: runs end-to-end with or without torch/sentencepiece, CPU-only, no Wikipedia download.

---

## Module Conformance (contracts.py)

```
m12 TransformerSummarizer (REAL module, consumed_by: [15, 16]):           provides:
  train(src_texts, tgt_texts, epochs=10, d_model=128, nhead=4) -> None      ✓
  summarize(text: str, max_length=60) -> str                                ✓  (greedy / extractive)
  rouge1(references, hypotheses) -> dict[str, float]                        ✓  {"precision","recall","f1"}
  save(path: str) -> None  /  load(path: str) -> None                       ✓  (pickle; torch state_dict or df)
```
Unlike m09/m11 (pedagogical), m12 is a **real pipeline module**: save/load present and wired into
m15 (agent tool `summarize_text`) and Day 16. rouge1 returns a **dict** (not a float like m11's bleu).

---

## Deviation from course_map.yaml

course_map Day 13 `corpus_subset: uz_wiki_summ` (online Wikipedia uz lead-paragraph, CONFIRMED
CC-BY-SA 3.0 in LICENSES.md). The **OFFLINE_FALLBACK** uses a small **original** uz maqola-xulosa corpus
(`uz_wiki_summ_mini.txt`, 16 pairs; Wikipedia dump not bundled). Local run is **CPU-only** (GPU is an
accelerator). Demo uses d_model=64 / 20 epochs for CPU speed; full-scale (d_model=128, T4, more epochs)
noted in a code comment. sentencepiece BPE simplified to word-level tokenization locally (sentencepiece
absent); ROUGE is demo-quality, stated honestly.

---

## Pending

- Full Kaggle kernel run with real Wikipedia uz pairs on GPU (torch `nn.Transformer` + sentencepiece BPE
  + ROUGE) — confirmed when notebooks are published as a Kaggle Dataset (Day 13, 6-iyul-2026).
- **L13** (Day 13 lecture — Transfer Learning, BERT/T5) is the next chronological artifact.
- m12 will be consumed by m15 (agent `summarize_text` tool, P15) and Day 16 pipeline.
