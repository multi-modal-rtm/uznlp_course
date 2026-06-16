# QA Report: Day 8 Practice (P7)

**Artifacts**: `course/practices/d08_p7_rnn.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m07_rnn_classifier.py`
**Bundled data**: `course/practices/d03_checkpoints/uz_sentiment_mini.txt` (reused; 118 labeled after dropping neutral)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L7 — Rekurrent neyron tarmoqlar (RNN)
**Next**: L8 (Day 8 lecture — GRU/LSTM)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `json.load` OK; nbformat 4.5; 28 cells (13 code, 15 md); all `id` |
| JSON valid — solutions | **PASS** | `json.load` OK; nbformat 4.5; 28 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True)** | **PASS (local)** | SOLUTIONS exec'd in sequence on Python 3.13.14 — **8/8 asserts passed**, zero exceptions. **Both paths verified**: torch 2.10.0+cpu (real `nn.RNN`) AND forced-numpy fallback. |
| Student stub cells compile | **PASS** | All 13 code cells `compile()` clean |
| **Locked assert (RNN step)** | **PASS** | §4A `h_1 = tanh([1,0]) = [0.762, 0.000]` (pure-numpy, torch-independent) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L7 [I2]-slayd bilan solishtiring` (§4A) |
| Every blanked region has paired assert | **PASS** | 3 blanked cells (§4B fit, §4C predict_proba, §4D eval+save/load) → paired asserts |
| m07 contract conformance | **PASS** | `fit(texts,labels,epochs,hidden_size,lr)` / `predict->str` / `predict_proba->dict` / `save` / `load` exact (contracts.py) |
| Capstone continuity (uses m01) | **PASS** | tokenization via `TextPreprocessor.preprocess` (m01) |
| Locked labels | **PASS** | only `ijobiy`/`salbiy` (no musbat/manfiy) |
| No GPU / VRAM | **PASS** | CPU-only; small RNN (hidden=64); VRAM peak 0 GB |
| Data size | **PASS** | bundled sentiment mini ≈ 12 KB (≪ 500 MB; real uz_news_mini online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads sentiment data) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` across all artifacts |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | 2 stray Cyrillic letters (`ка`→`ka`, `га`→`ga`) found and fixed; re-verified clean; SOLUTIONS re-exec'd (8/8 asserts) after fix |

**Overall: ALL GATES PASS**

---

## torch was present locally — both paths verified

The HOLAT_HISOBOT premise was "no local torch", but the local env actually has
**torch 2.10.0+cpu**. This is favorable: the **real PyTorch `nn.RNN` path** was verified
end-to-end locally (not just the fallback). The torch-optional design (`HAS_TORCH` flag)
is retained for laptop/portability robustness, and the numpy fallback was **also** verified
by forcing `HAS_TORCH=False`.

| Path | Train accuracy | predict(clear pos / neg) | predict_proba sum |
|---|---|---|---|
| torch `nn.RNN` (hidden=64, 15 ep, lr=0.01) | **0.89** (F1≈0.9) | ijobiy / salbiy ✓ | 1.000 |
| pure-numpy RNN + BPTT (forced) | **0.76** | ijobiy / salbiy ✓ | 1.000 |

The locked §4A `h_1` assert is pure-numpy and path-independent.

---

## Locked / Verified Numbers

| Assert | Slide | Expected | Computed |
|--------|-------|----------|----------|
| §4A RNN step `h_1` | L7 [I2] | `[0.762, 0.000]` | `[0.7616, 0.0]` |

The §4A cell reproduces the lecture hand example (`tanh(W_h·0 + W_x·[1,0])`) — **P7's
first assert**, matching course_map Day 8 paired-lecture L7 `hand_example`.

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L7 [C], timing |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, m01 path, HAS_TORCH |
| §2 Yaxlit natija | 2 | MD+Code | OK — finished RNN + F1 + predict demo |
| §3 PRIMM periferiya | 4 | Mixed | OK — m01→sequences+padding; DataLoader/collate (torch-guarded); Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked) + 3 blanked (4B fit, 4C proba, 4D eval+save/load) each with paired assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m07 import/contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — RNN end-sensitivity / SOV / vanishing gradient, exit ticket |

Total: 28 cells (13 code, 15 markdown). Blanked core cells: 3, each paired with an assert.

---

## torch-Optional Design (m03/m06 pattern extended)

`torch` may be absent. m07 and the notebook branch on `HAS_TORCH`:
- **Kaggle path**: `nn.Embedding + nn.RNN(nonlinearity="tanh") + nn.Linear`,
  `CrossEntropyLoss` + `Adam`, padded batches.
- **Offline path** (forced/torchless): pure-numpy RNN classifier — forward through the
  `tanh` recurrence + manual BPTT/SGD; embedding matrix + `W_h`/`W_x`/`W_o`.

Result: the notebook runs end-to-end with or without torch, and without uz_news_full —
mirroring the gensim/datasketch/nltk-optional pattern of P3/P4/P5/P6.

---

## Module Conformance (contracts.py)

```
m07 RNNClassifier:                                            provides:
  fit(texts, labels, epochs=5, hidden_size=64, lr=1e-3) -> None   ✓
  predict(text: str) -> str            ('ijobiy'/'salbiy')        ✓
  predict_proba(text: str) -> dict[str, float]                    ✓
  save(path: str) / load(path: str)                               ✓
```
Uses m01 `TextPreprocessor` for tokenization. `consumed_by: [9, 16]`
(m08 baseline comparison, Day 16 pipeline).

---

## Deviation from course_map.yaml

course_map Day 8 `corpus_subset: uz_news_mini` (5000-sample, online, binarized as sentiment
proxy). The **OFFLINE_FALLBACK** uses the bundled `uz_sentiment_mini.txt` (real
ijobiy/salbiy labels) — consistent with m07's role as a sentiment classifier compared
against m02 (LogReg/NB) and m08 (GRU/LSTM) in w3. Demo uses `epochs=15, lr=0.01` on the
tiny offline set; full-scale `epochs=5` on uz_news_mini (5000) noted per kaggle-hardware
(SimpleRNN, hidden=64 ≈ 2–3 min CPU).

---

## Pending

- Full Kaggle kernel run with real `uz_news_mini` (5000) on the torch path — confirmed
  when notebooks are published as a Kaggle Dataset (Day 8, 26-iyun-2026).
- **L8** (Day 8 lecture — GRU/LSTM) is the next chronological artifact.
