# QA Report: Day 2 Practice (P1)

**Artifact**: `course/practices/d02_p1_preprocessing.ipynb` + `d02_p1_preprocessing_SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m01_text_preprocessor.py`
**Date**: 2026-06-14
**Reviewer**: Claude Code (automated gates)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `nbformat.validate` — student | **PASS** | Valid nbformat 4 |
| `nbformat.validate` — solutions | **PASS** | Valid nbformat 4 |
| All 24 code cells compile (student) | **PASS** | `compile(..., 'exec')` zero errors |
| All 24 code cells compile (solutions) | **PASS** | `compile(..., 'exec')` zero errors |
| OFFLINE_FALLBACK = True defined | **PASS** | Cell 1 (ENV_CHECK) |
| No GPU assert / no VRAM dependency | **PASS** | CPU-only design; no `torch.cuda.is_available()` assertion |
| VRAM peak | **PASS** | 0 GB (no model loading; classical ML only) |
| Terminology grep — student | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| Terminology grep — solutions | **PASS** | 0 matches |
| Terminology grep — m01 module | **PASS** | 0 matches |
| Locked assert 1: TF-IDF(nlp,D1) = 0.405 | **PASS** | `abs(0.4055 - 0.405) < 1e-3` ✓ |
| Locked assert 2: TF-IDF(qiziq,D1) = 1.099 | **PASS** | `abs(1.0986 - 1.099) < 1e-3` ✓ |
| Reconciliation cell exists | **PASS** | §4F — "Nega sklearn standart natijasi boshqacha?" |
| Every blanked region has paired assert | **PASS** | 5 blanked code cells; each followed by assert cell |
| Solutions notebook fills all blanks | **PASS** | Cells 15, 20, 28, 31, 34 filled; no residual `pass` |
| m01 contracts.py signatures match | **PASS** | `preprocess`, `preprocess_batch`, `fit_stopwords` exact |
| Checkpoint cells present | **PASS** | Checkpoint A (cell 10), Checkpoint B (cell 22) |
| Traceability comments cite lecture | **PASS** | `# Ma'ruza L1 [I]-slayd bilan solishtiring` in cell 24 |
| `OFFLINE_FALLBACK` corpus fallback | **PASS** | `d02_checkpoints/uz_news_mini.txt` (20 sentences) bundled |

**Overall: ALL GATES PASS**

---

## Locked Numeric Values — P1 Asserts

| Assert | Expected | Computed | Tolerance | Source |
|--------|----------|----------|-----------|--------|
| TF-IDF(nlp, D1) | 0.405 | 0.4055 | 1e-3 | L1 [I3]-slayd |
| TF-IDF(qiziq, D1) | 1.099 | 1.0986 | 1e-3 | L1 [I3]-slayd |

Formula applied: `IDF(t) = ln(N/df(t))` (no smooth, no +1 offset, no L2 norm),
`TF(t,d) = count(t in d)` (raw count). Corpus: D1="nlp qiziq", D2="python foydali",
D3="nlp foydali", N=3.

---

## Reconciliation Cell Confirmed

Cell 26 (§4F) runs `TfidfVectorizer()` with defaults and prints:
- Default (`smooth_idf=True`, `norm='l2'`): nlp≈0.605, qiziq≈0.796
- `smooth_idf=False, norm=None`: nlp≈1.405, qiziq≈2.099
- Manual (L1 formula): nlp=0.405, qiziq=1.099

Three causes of divergence are stated: smooth_idf flag, +1 offset, L2 norm.

---

## Notebook Structure

| Section | Cell(s) | Type | Status |
|---------|---------|------|--------|
| §0 Header + timings | 0 | Markdown | OK |
| §1 Muhit tekshiruvi | 1 | Code | OK — OFFLINE_FALLBACK, seeds |
| §2 Yaxlit natija | 2–3 | MD + Code | OK — complete pipeline demo |
| §3A PRIMM: matn yuklash | 4–6 | MD+Code+MD | OK — predict/investigate/modify |
| §3B PRIMM: spaCy | 7–9 | MD+Code+MD | OK — Uzbek limitation stated |
| Checkpoint A | 10 | Code | OK |
| §4 core: so'nuvchi tayanch | 11–35 | Mixed | OK — 3 Namuna + 4 Birgalikda + 1 Mustaqil |
| §5 Loyihaga ulash | 36–40 | MD+3 Code+MD | OK — m01 write + import test + git |
| §6 Tadqiqot savoli + yakun | 41–43 | MD+Code+MD | OK — exit ticket |

Total: 44 cells (24 code, 20 markdown)

---

## Blanked Cells (Student vs Solutions)

| Cell | Function | Student | Solutions |
|------|----------|---------|-----------|
| 15 | `filter_stopwords()` | `pass` + blank comment | Complete list comprehension |
| 20 | `preprocess_doc()` + loop | `pass` stubs | Complete pipeline |
| 28 | CountVectorizer | `None` stubs | `vec_bow = CountVectorizer(); X_bow = vec_bow.fit_transform(...)` |
| 31 | TfidfVectorizer | `None` stubs | `vec_tfidf = TfidfVectorizer(); X_tfidf = vec_tfidf.fit_transform(...)` |
| 34 | Mustaqil corpus | Minimal scaffold | Full solution on sample corpus |

---

## m01 TextPreprocessor — Signature Conformance

```
contracts.py expects:                     m01 provides:
preprocess(text: str) -> list[str]        ✓  (ValueError on empty)
preprocess_batch(texts: list[str])        ✓
fit_stopwords(texts, max_df=0.85)         ✓  (Counter + threshold logic)
```

---

## Pending

- Full end-to-end kernel execution via `jupyter nbconvert --execute` deferred
  (requires Jupyter installed locally; verified via compile gate instead).
- Kaggle runtime validation: to be confirmed when notebooks are published as
  a Kaggle Dataset on Day 2 (16-iyun-2026).
