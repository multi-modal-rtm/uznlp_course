# QA Report: Week 1 Milestone (w1)

**Artifacts**: `course/milestones/w1_milestone.md` (brief) + `course/milestones/w1_check.py` (self-check)
**Date**: 2026-06-15
**Reviewer**: Claude Code (automated gates + local execution)
**course_map**: `id: w1` — "1-hafta milestone: TextPreprocessor integratsiyasi"
(date 17-iyun, deadline 22-iyun; `modules_covered: [1]`, `modules_preview: [2]`)

---

## Scope (per course_map id: w1)

Two-phase, by design (only m01 is built by Wed 17-iyun; m02 is built Thu/P2):
- **Phase A** (Wed, async): apply **m01 TextPreprocessor** to the tinglovchi's own
  50–100 doc corpus → vocab size, token/stopword ratio, top-20 words.
- **Phase B** (Thu–weekend): **m01 → m02 pipeline** (preprocess → TF-IDF → sentiment predict).
- Written reflection (3–5 sentences).

**m03/m04 are NOT in scope** — they integrate in w2 (24-iyun). Confirmed against course_map.

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| **w1_check.py local execution** | **PASS** | Python 3.13.14, numpy 2.2.6, sklearn 1.8.0 — **21/21 checks passed**, exit 0, no gensim/datasketch |
| Phase A — m01 contract | **PASS** | `preprocess`/`preprocess_batch`/`fit_stopwords` present; lowercase; **ValueError on empty**; batch returns list-of-lists |
| Phase A — corpus stats | **PASS** | uz_news_mini (20 docs): vocab 125, tokens 156→145 (ratio 0.93), top-5 printed; vocab < raw tokens |
| Phase B — m02 contract | **PASS** | `fit`/`predict`/`predict_proba`/`save`/`load` present |
| Phase B — m01→m02 pipeline | **PASS** | binarize→{ijobiy,salbiy}; pos→`ijobiy`, neg→`salbiy`; proba sums to 1; save/load round-trip |
| Locked labels | **PASS** | only `ijobiy`/`salbiy` used (no musbat/manfiy) |
| Terminology grep (both files) | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| Brief: tinglovchi-facing, Uzbek, sentence-case | **PASS** | "tinglovchi/siz", Phase A/B tasks, deadline 22-iyun, reflection, submission |
| Scope correctness (m01/m02 only) | **PASS** | brief + check exclude m03/m04; note points to w2 |
| No GPU / VRAM | **PASS** | CPU-only (numpy/sklearn) |

**Overall: ALL GATES PASS**

---

## w1_check.py — Local Run Output (21/21)

```
[Faza A] m01 TextPreprocessor — shartnoma mosligi va korpus statistikasi
  ✓ m01.preprocess / preprocess_batch / fit_stopwords mavjud
  ✓ preprocess() list; tokenlar kichik harfda; bo'sh matnda ValueError
  ✓ korpus to'liq qayta ishlandi; lug'at xom matndan kichik
    Korpus: 20 hujjat | lug'at hajmi: 125 so'z | tokenlar 156 -> 145 (0.93)
[Faza B] m01 -> m02 pipeline — SentimentClassifier
  ✓ m02.fit / predict / predict_proba / save / load mavjud
  ✓ binarizatsiya ijobiy/salbiy; pos->ijobiy; neg->salbiy
  ✓ predict_proba {ijobiy,salbiy}, yig'indi 1; save/load mos
NATIJA: 21 tekshiruvning hammasi O'TDI ✓
```

---

## Notes

- **Self-check paths** are computed relative to the script (`course/milestones/w1_check.py`
  → repo root), using bundled `d02_checkpoints/uz_news_mini.txt` (Phase A) and
  `d03_checkpoints/uz_sentiment_mini.txt` (Phase B). Runs offline; no external data/libs.
- **Phase A corpus**: brief asks the tinglovchi to bring their OWN 50–100 docs; the
  bundled mini-corpus is the fallback the self-check uses.
- **m01→m02 continuity**: m02 internally composes m01 (`TextPreprocessor`), so Phase B
  exercises the integration inherently.
- **Deviation from course_map**: none. Scope (m01 + m02 preview) matches `id: w1` exactly.

## Pending

- Tinglovchi-side submission (own corpus stats + reflection) — collected by maintainer
  at deadline (22-iyun), per the brief.
- **w2 milestone** (24-iyun) integrates m01–m05 (incl. m03/m04) — next milestone.
