# QA Report: Day 3 Practice (P2)

**Artifact**: `course/practices/d03_p2_sentiment.ipynb` + `d03_p2_sentiment_SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m02_sentiment_classifier.py`
**Bundled data**: `course/practices/d03_checkpoints/uz_sentiment_mini.txt`
**Date**: 2026-06-15
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L2 — Klassik tasnif, metrikalar, etika
**Next practice bridge**: P3 (so'z embeddinglari, m03)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `JSON.parse` OK; nbformat 4.5; 41 cells (23 code, 18 md); all cells have `id` |
| JSON valid — solutions | **PASS** | `JSON.parse` OK; nbformat 4.5; 41 cells; all ids present |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True)** | **PASS (local)** | Solutions notebook executed cell-by-cell on Python 3.13.14 / sklearn 1.8.0 / numpy 2.2.6 — **every assert passed**, zero exceptions |
| Student stub cells compile | **PASS** | All 23 code cells `compile()` clean (blanked stubs syntactically valid) |
| Locked assert 1: NB nisbat = 3.375 → ijobiy | **PASS** | §4A hand calc = 3.375; `result == "ijobiy"`; sklearn MultinomialNB confirms ratio 3.375 |
| Locked assert 2: metrics A=0.88, F1=0.50 | **PASS** | §4D confusion-matrix hand calc reproduces L2 [I3] |
| Traceability comments cite lecture | **PASS** | `# Ma'ruza L2 [I2]-slayd` (§4A), `# Ma'ruza L2 [I3]-slayd` (§4D) |
| Every blanked region has paired assert | **PASS** | 5 blanked cells (§4B,§4C,§4E,§4F,§4G) → 5 paired assert cells |
| Solutions fills all blanks | **PASS** | All 5 solution cells execute; no residual `None`/`pass` reaching asserts |
| m02 contract conformance | **PASS** | `fit/predict/predict_proba/save/load` match `contracts.py`; round-trip OK |
| Capstone continuity (uses m01) | **PASS** | §3B + m02 import & use `TextPreprocessor` (m01) |
| No GPU / VRAM dependency | **PASS** | CPU-only classical ML; VRAM peak 0 GB; no `torch.cuda` assert |
| Data size | **PASS** | bundled `uz_sentiment_mini.txt` ≈ 11 KB (≪ 500 MB) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `random_state=42` in split |
| Checkpoint cells present | **PASS** | Checkpoint A (§3B split save/load) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` across all 4 artifacts |
| pdflatex | n/a | (practice notebook, no LaTeX) |

**Overall: ALL GATES PASS**

---

## Locked Numeric Values — P2 Asserts

| Assert | Slide | Expected | Computed | Tolerance |
|--------|-------|----------|----------|-----------|
| NB posterior nisbat (§4A) | L2 [I2] | 3.375 | 3.375 | 0.01 |
| NB bashorat (§4A) | L2 [I2] | `ijobiy` | `ijobiy` | exact |
| sklearn MultinomialNB nisbat (§4A) | L2 [I2] | ≈3.375 | 3.375 | 0.05 |
| accuracy (§4D) | L2 [I3] | 0.88 | 0.880 | 0.01 |
| precision (§4D) | L2 [I3] | 0.4286 | 0.429 | 0.01 |
| recall (§4D) | L2 [I3] | 0.60 | 0.600 | 0.01 |
| F1 (§4D) | L2 [I3] | 0.50 | 0.500 | 0.01 |

Hand-worked NB corpus (verbatim from L2 [I2] / course_map hand_example):
`pos=['yaxshi film','ajoyib film']`, `neg=['yomon film']`, `|V|=4`, Laplace α=1,
test `'yaxshi film'` → `score_pos=1/16`, `score_neg=1/54`, `nisbat=27/8=3.375`.

---

## Trained-Model Results (offline bundled sample, seed 42)

Deterministic on the bundled `uz_sentiment_mini.txt` (118 labeled reviews after
binarization; 88 train / 30 test, stratified):

| Model | Test accuracy | F1 (ijobiy) | Misclassified |
|-------|---------------|-------------|---------------|
| LogisticRegression | 0.967 | 0.968 | 1 / 30 |
| MultinomialNB (α=1.0) | 0.967 | 0.968 | 1 / 30 |

(High scores reflect the clean, polarized bundled sample; the real HF dataset
loaded online is noisier — notebook states this.)

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header + timings | 1 | MD | OK |
| §1 Muhit tekshiruvi | 1 | Code | OK — seeds, OFFLINE_FALLBACK, m01 path |
| §2 Yaxlit natija | 2 | MD+Code | OK — full inline sentiment pipeline |
| §3A PRIMM: load + binarize | 3 | MD+Code+MD | OK — predict/investigate/modify |
| §3B PRIMM: m01 + split + TF-IDF | 3 | MD+Code+MD | OK — uses TextPreprocessor (m01) |
| Checkpoint A | 1 | Code | OK |
| §4 core (so'nuvchi tayanch) | 16 | Mixed | OK — 3 Namuna (4A,4A2,4D) + 5 Birgalikda blanks (4B,4C,4E,4F) + 1 Mustaqil (4G) |
| §5 Loyihaga ulash | 5 | MD+Code | OK — write m02, import+contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — LR vs NB, exit ticket |

Total: 41 cells (23 code, 18 markdown). Blanked core cells: 5, each paired with an assert.

---

## m02 SentimentClassifier — Signature Conformance

```
contracts.py expects:                          m02 provides:
fit(texts: list[str], labels: list[str])       ✓  (ValueError on length mismatch)
predict(text: str) -> str                      ✓  ('ijobiy' / 'salbiy')
predict_proba(text: str) -> dict[str,float]    ✓  (keys {ijobiy,salbiy}, sum=1)
save(path: str) -> None                        ✓  (pickle: vec+clf+state)
load(path: str) -> None                        ✓  (round-trip verified)
```
`__init__(model='logreg'|'nb')` added (not constrained by contract). Internally
composes m01 `TextPreprocessor` → `TfidfVectorizer` → classifier.

---

## Judgment Call — Corpus Deviation from course_map.yaml (needs human note)

`course_map.yaml` Day 3 lists `corpus_subset: uz_news_mini` (binarize 2 topic
classes as a sentiment **proxy**). This P2 was produced — per the explicit task
instruction — on **`uz_sentiment_uzum`** (Uzum Market reviews, real sentiment
labels, MIT, `risqaliyevds/uzbek-sentiment-analysis`).

**Rationale (recommend approving + updating the map):**
- Real ijobiy/salbiy labels are pedagogically far stronger than a topic proxy.
- Same dataset is used in P13 (Day 14, BERT fine-tuning) → enables a clean
  classical (m02) vs transformer (m13) comparison on identical data.
- License is **CONFIRMED (MIT)** in `LICENSES.md` (`uz_sentiment_uzum`).

**Action item for maintainer:** update `course_map.yaml` Day 3 `corpus_subset`
to `uz_sentiment_uzum` to match this artifact.

---

## Offline Data Note

`d03_checkpoints/uz_sentiment_mini.txt` is a **small representative sample**
(126 rows: 60 rating-{4,5}, 58 rating-{1,2}, 8 rating-3) authored in realistic
Uzbek e-commerce review style, bundled for `OFFLINE_FALLBACK=True` execution —
mirroring how P1 bundled `uz_news_mini.txt`. The **online path** (`OFFLINE_FALLBACK=False`)
loads the real HF dataset via `load_dataset("risqaliyevds/uzbek-sentiment-analysis")`.
Format: `rating<TAB>text`; the same `binarize_rating()` runs on both paths.

---

## Pending

- Full Kaggle kernel run with the **real** HF dataset (not the offline sample)
  and matplotlib confusion-matrix display — to be confirmed when notebooks are
  published as a Kaggle Dataset (Day 3, 18-iyun-2026).
- `course_map.yaml` Day 3 corpus update (see judgment call above).
