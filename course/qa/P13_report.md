# QA Report: Day 14 Practice (P13)

**Artifacts**: `course/practices/d14_p13_finetune.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m13_bert_classifier.py` (REAL module — `consumed_by`: m15 agent, app.py FastAPI)
**Bundled data**: `course/practices/d14_checkpoints/uz_sentiment_mini.csv` (original uz sentiment, 24 rows: 12 ijobiy + 12 salbiy)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L13 — Transfer Learning va oldindan o'qitilgan modellar (BERT, T5)
**Next**: L14 (Day 14 lecture)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `nbformat.validate` OK; nbformat 4.5; 30 cells (14 code, 16 md); all `id` |
| JSON valid — solutions | **PASS** | `nbformat.validate` OK; nbformat 4.5; 30 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True, USE_TRANSFORMERS=False, CPU)** | **PASS (local)** | SOLUTIONS exec'd in sequence on Python 3.13.14 — **12/12 asserts passed**, zero exceptions. Classical TF-IDF+LogReg fallback (m02 pattern) executed; pure-torch BCE assert independent. |
| Student stub cells compile | **PASS** | All 14 code cells `compile()` clean |
| **Locked assert (BCE / sigmoid)** | **PASS** | §4A `σ(2.0)=0.880`, `BCE=nn.BCEWithLogitsLoss()(2.0, 1.0)=0.128` (pure-torch, transformers-independent) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L13 [I1]-slayd` (§4A) |
| Every blanked region has paired assert | **PASS** | 3 blanked cells (§4B hparams, §4C m13.fit, §4D — full but asserted) → paired asserts |
| m13 contract conformance | **PASS** | `fit(texts,labels,model_name,epochs,batch_size,lr)` / `predict->str` / `predict_proba->dict` / `save` / `load` exact (contracts.py) |
| **Labels locked `ijobiy`/`salbiy`** | **PASS** | predict ∈ {`ijobiy`,`salbiy`}; predict_proba keys exactly `{ijobiy, salbiy}`; **0 `musbat`/`manfiy`** anywhere |
| **save/load test (real module)** | **PASS** | §5 `save → load → predict` verified (classical: pickle; HF path: `save_pretrained` dir — code present) |
| predict / predict_proba structural | **PASS** | predict returns str ∈ labels; predict_proba 2-key dict, each ∈ [0,1], sum ≈ 1 (not exact/high-F1 — small data, demo-quality, honest) |
| No GPU / VRAM | **PASS** | CPU-only; LogReg fallback; VRAM peak 0 GB |
| Data size | **PASS** | bundled corpus ≈ 1.3 KB (≪ 500 MB; real Uzum 5000 subsample online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads sentiment corpus) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi`; 0 `musbat\|manfiy` (all 4 artifacts) |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | all artifacts verified clean (notebooks, module, corpus) |

**Overall: ALL GATES PASS**

---

## transformers present locally, but fine-tuning is Kaggle-only — fallback verified

Local env has `transformers` importable **but**: (a) real DistilBERT `from_pretrained` needs a ~500 MB
download (internet); (b) CPU fine-tuning is slow; (c) `Trainer` additionally requires `accelerate>=0.26`
(absent). So the notebook forces `USE_TRANSFORMERS = HAS_TRANSFORMERS and not OFFLINE_FALLBACK` →
**False locally** → the **classical TF-IDF + LogisticRegression** fallback (m02 pattern) runs and is
verified end-to-end. The real DistilBERT + `Trainer` path is shown as runnable Kaggle code (commented /
guarded) and executes there (GPU + internet).

| Path | fit | predict("mahsulot juda sifatli va arzon") |
|---|---|---|
| classical TF-IDF + LogReg (local, forced) | instant | **"ijobiy"** ✓ |
| DistilBERT + Trainer (Kaggle, USE_TRANSFORMERS=True) | ~939 steps (5000, b16, 3 ep) | real fine-tuned BERT |

The locked §4A BCE is pure-torch and path-independent. §4B teaches the Trainer hyperparameters as a
dict (path-independent — avoids the `accelerate` dependency that `TrainingArguments()` construction needs).

---

## Locked / Verified Numbers

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I1]** | `σ(2.0)`, BCE (y=1) | **0.880, 0.128** | **P13 first assert** (course_map lock) |

§4A reproduces the lecture's sigmoid/BCE — **P13's first assert** — matching course_map Day 14
paired-lecture L13 `hand_example` (`# Ma'ruza L13 [I1]-slayd`).

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L13 [C], timing, MIT license + ijobiy/salbiy lock disclosure |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, HAS_TORCH/TRANSFORMERS/DATASETS, USE_TRANSFORMERS, module path |
| §2 Yaxlit natija | 2 | MD+Code | OK — load_sentiment + m13.fit + predict + predict_proba demo |
| §3 PRIMM periferiya | 3 | Mixed | OK — binarize(rating)→ijobiy/salbiy + HF tokenizer (Kaggle, commented); Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked BCE) + 4B hparams + 4C m13 fit/predict + 4D predict_proba, each blanked region + assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m13 contract test, save/load test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — fallback accuracy demo; mBERT WordPiece; exit ticket |

Total: 30 cells (14 code, 16 markdown). Blanked core cells: 3 (§4B hparams, §4C fit), each paired with an assert; §4D mustaqil + structural assert.

---

## transformers-Optional + datasets-Optional Design

`transformers`/`datasets` may be absent or undesirable locally. m13 branches on `USE_TRANSFORMERS`:
- **Kaggle path** (`USE_TRANSFORMERS=True`, GPU+internet): `AutoModelForSequenceClassification`
  (DistilBERT, `num_labels=2`) + `Trainer` (lr=2e-5, batch=16, epochs=3, warmup=100); predict/proba via logits→softmax.
- **Offline path** (forced `USE_TRANSFORMERS=False`): `TfidfVectorizer` + `LogisticRegression` (m02 pattern).
- **datasets** absent → corpus loaded from bundled CSV (not `load_dataset`).
- Labels locked `ijobiy`/`salbiy` on both paths (`_LAB2I = {salbiy:0, ijobiy:1}`).

Result: runs end-to-end with or without transformers/datasets/GPU, CPU-only, no model download.

---

## Module Conformance (contracts.py)

```
m13 FineTunedClassifier (REAL module, consumed_by: m15 (P15), app.py (M4 P16)):       provides:
  fit(texts, labels, model_name="distilbert-base-multilingual-cased",
      epochs=3, batch_size=16, lr=2e-5) -> None                                        ✓
  predict(text: str) -> str                                                            ✓  ("ijobiy"/"salbiy")
  predict_proba(text: str) -> dict[str, float]                                         ✓  {"ijobiy","salbiy"}
  save(path: str) -> None  /  load(path: str) -> None                                  ✓  (HF save_pretrained dir / pickle file)
```
Real pipeline module: save/load present and wired into m15 (`sentiment_classify`) and `app.py` (FastAPI).

---

## Deviation from course_map.yaml

course_map Day 14 `corpus_subset: uz_sentiment_uzum` (`risqaliyevds/uzbek-sentiment-analysis`, MIT,
CONFIRMED in LICENSES.md; 5000 subsample online). The **OFFLINE_FALLBACK** uses a small **original** uz
sentiment corpus (`uz_sentiment_mini.csv`, 24 rows balanced). Local run is **CPU-only** with the classical
fallback (transformers fine-tuning needs download + `accelerate`, Kaggle-only). Trainer hyperparameters
taught via a dict (path-independent). Accuracy is demo-quality, stated honestly.

---

## Pending

- Full Kaggle kernel run with real Uzum dataset on GPU (DistilBERT + `Trainer` + `accelerate`) — confirmed
  when notebooks are published as a Kaggle Dataset (Day 14, 7-iyul-2026).
- **L14** (Day 14 lecture) is the next chronological artifact.
- m13 will be consumed by m15 (agent `sentiment_classify` tool, P15) and `app.py` (FastAPI, M4 P16).
