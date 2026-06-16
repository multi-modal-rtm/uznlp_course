# QA Report: Day 9 Practice (P8)

**Artifacts**: `course/practices/d09_p8_gru_lstm.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m08_gru_lstm_classifier.py`
**Bundled data**: `course/practices/d03_checkpoints/uz_sentiment_mini.txt` (reused; 118 labeled)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L8 — GRU va LSTM
**Next**: L9 (Day 9 lecture — text generation / Bi-RNN)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `json.load` OK; nbformat 4.5; 28 cells (13 code, 15 md); all `id` |
| JSON valid — solutions | **PASS** | `json.load` OK; nbformat 4.5; 28 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True, CPU)** | **PASS (local)** | SOLUTIONS exec'd in sequence on Python 3.13.14 — **8/8 asserts passed**, zero exceptions. **Both paths verified**: torch 2.10+cpu (real `nn.LSTM`/`nn.GRU`, num_layers=2) AND forced-numpy reservoir fallback. |
| Student stub cells compile | **PASS** | All 13 code cells `compile()` clean |
| **Locked assert (forget gate)** | **PASS** | §4A `f_t = σ(1.5) = 0.818` (pure-numpy, torch-independent) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L8 [I3]-slayd bilan solishtiring` (§4A) |
| Every blanked region has paired assert | **PASS** | 3 blanked cells (§4B fit, §4C compare_report, §4D vanishing+save/load) → paired asserts |
| m08 contract conformance | **PASS** | `fit(texts,labels,arch,epochs,hidden_size,num_layers,lr)` / `predict->str` / `compare_report->dict` / `save` / `load` exact (contracts.py) |
| compare_report design | **PASS** | returns `{'lstm':{f1,accuracy,inference_time}, 'gru':{...}}` — trains both archs on cached fit data |
| Capstone continuity (uses m01, m06) | **PASS** | tokenization via m01; §3 builds pretrained Embedding from m06 (CustomWord2Vec) |
| Locked labels | **PASS** | only `ijobiy`/`salbiy` (no musbat/manfiy) |
| No GPU / VRAM | **PASS** | CPU-only locally (torch cpu); GPU is accelerator-only per kaggle-hardware |
| Data size | **PASS** | bundled sentiment mini ≈ 12 KB (≪ 500 MB; real uz_news_full online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, reservoir `RandomState(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads sentiment data) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` across all artifacts |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | all three artifacts verified clean (Cyrillic-lookalike trap checked → 0) |

**Overall: ALL GATES PASS**

---

## torch present locally — both paths verified

Local env has **torch 2.10.0+cpu**, so the real PyTorch `nn.LSTM`/`nn.GRU`
(num_layers=2) path was verified end-to-end on CPU. The torch-optional fallback
(forced `HAS_TORCH=False`) was **also** verified.

| Path | active LSTM acc | compare_report (lstm / gru F1) | predict(neg) |
|---|---|---|---|
| torch `nn.LSTM`/`nn.GRU` (hidden=32, 2 layers, 8 ep) | 0.94 | 0.97 / 0.98 | salbiy ✓ |
| numpy reservoir + LogReg readout (forced) | 0.85 | 0.85 / 0.83 | salbiy ✓ |

The locked §4A `f_t` assert is pure-numpy and path-independent.

---

## Locked / Verified Numbers

| Assert | Slide | Expected | Computed |
|--------|-------|----------|----------|
| §4A forget gate `f_t` | L8 [I3] | `0.818` | `0.8176` |
| §4D vanishing demo | L8 [I2] | `0.9^15 > 0.42^15` | `0.206 > 5.6e-6` ✓ |

The §4A cell reproduces the lecture forget-gate hand example — **P8's first assert**,
matching course_map Day 9 paired-lecture L8 `hand_example`.

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L8 [C], timing |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, m01 path, HAS_TORCH |
| §2 Yaxlit natija | 2 | MD+Code | OK — finished compare_report (GRU vs LSTM) |
| §3 PRIMM periferiya | 4 | Mixed | OK — m06 pretrained Embedding matrix; learning-curve helper; Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked) + 3 blanked (4B fit, 4C compare_report, 4D vanishing+save/load) each with paired assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m08 import/contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — GRU vs LSTM trade-off, exit ticket |

Total: 28 cells (13 code, 15 markdown). Blanked core cells: 3, each paired with an assert.

---

## torch-Optional Design (m07 pattern extended)

`torch` may be absent. m08 branches on `HAS_TORCH`:
- **Kaggle path**: `nn.Embedding + nn.LSTM/nn.GRU(num_layers=2) + nn.Linear`,
  `CrossEntropyLoss` + `Adam` (full BPTT via autograd).
- **Offline path** (forced/torchless): random-init GRU/LSTM **forward** (reservoir) →
  last hidden state `h_T` → sklearn `LogisticRegression` readout. Real gated forward
  dynamics; convex readout (no fragile hand-written BPTT). `compare_report` runs the
  reservoir for both archs.

Result: end-to-end with or without torch, CPU-only, no uz_news_full needed —
mirroring the gensim/datasketch/nltk/torch-optional pattern of P3–P7.

---

## Module Conformance (contracts.py)

```
m08 GRULSTMClassifier:                                                       provides:
  fit(texts, labels, arch='lstm', epochs=10, hidden_size=128,
      num_layers=2, lr=1e-3) -> None                                          ✓
  predict(text: str) -> str            ('ijobiy'/'salbiy')                    ✓
  compare_report() -> dict             {'lstm':{...}, 'gru':{...}}            ✓
  save(path) / load(path)                                                     ✓
```
Uses m01 (tokenization) and m06 (pretrained Embedding, §3 periphery).
`consumed_by: [14, 16]`.

---

## Deviation from course_map.yaml

course_map Day 9 `corpus_subset: uz_news_full` (online) and `gpu_required: true`.
The **OFFLINE_FALLBACK** uses bundled `uz_sentiment_mini.txt` (ijobiy/salbiy) — same as
P7, consistent with m08's role (compared vs m07/m02 in w3). Local run is **CPU-only**
(GPU is an accelerator, not a requirement, per kaggle-hardware). Demo uses
`hidden=32, num_layers=2, epochs=8`; full-scale `hidden=128, epochs=10` on uz_news_full
(GPU) noted in a comment.

---

## Pending

- Full Kaggle kernel run with real `uz_news_full` on GPU (torch path) — confirmed when
  notebooks are published as a Kaggle Dataset (Day 9, 29-iyun-2026).
- **L9** (Day 9 lecture — text generation / Bi-RNN) is the next chronological artifact.
- **w3 milestone** (1-iyul) integrates m01–m08.
