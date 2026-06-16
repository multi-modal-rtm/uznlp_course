# QA Report: Week 2 Milestone (w2)

**Artifacts**: `course/milestones/w2_milestone.md` (brief) + `course/milestones/w2_check.py` (self-check)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**course_map**: `id: w2` — "2-hafta milestone: Klassik pipeline integratsiyasi"
(date 24-iyun, deadline 29-iyun; `modules_covered: [1,2,3,4,5]`)

---

## Scope (per course_map id: w2)

One full integration (all of m01–m05 are built by Day 6, so no two-phase split):

- **Integration 1 — m01 → m04**: spell-correct (`correct`) → LSH retrieve
  (`retrieve_lsh`) → top-k documents.
- **Integration 2 — m01 + m02**: preprocess → TF-IDF → `predict` (sentiment,
  `ijobiy`/`salbiy`).
- **Integration 3 — m05**: autocomplete next-word suggestion (`complete`).
- **Coverage extra — m03**: `embed` / `most_similar` / `oov_rate` (functional
  check; `modules_covered` includes module 3 even though the `capstone_integration`
  prose emphasizes m01/m02/m04/m05).

**m05b (POSTagger) and m06 (CustomWord2Vec) are NOT in scope** — m05b is
pedagogical (`consumed_by: []`); m06 is Day 7. Confirmed against course_map.

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| **w2_check.py local execution** | **PASS** | Python 3.13.14, numpy 2.2.6, sklearn 1.8.0 — **40/40 checks passed**, exit 0, no gensim/datasketch/nltk |
| Contract conformance m01–m05 | **PASS** | All `contracts.py` methods present for m01, m02, m03, m04, m05 |
| Integration 1 (m01→m04) | **PASS** | `edit_distance` L4-locked (`qo'l/ko'l`=1, `dastur/dastir`=1); typo `telifon`→`telefon`; `retrieve_lsh` returns top-3 docs; save/load round-trip |
| Integration 2 (m01+m02) | **PASS** | binarize→{ijobiy,salbiy}; pos→`ijobiy`, neg→`salbiy`; proba sums to 1 |
| Integration 3 (m05) | **PASS** | `complete("yangi",3)` returns in-vocab words; `perplexity` finite positive (20.29) |
| m03 functional | **PASS** | `embed` shape (50,); OOV→zero-vector; `most_similar("toshkent")` city cluster; `oov_rate`=0.50 |
| Locked labels | **PASS** | only `ijobiy`/`salbiy` used as sentiment (the two `musbat`/`manfiy` hits are "musbat son"=positive number, and one explicit "musbat/manfiy emas" prohibition note) |
| Terminology grep (both files) | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| Brief: tinglovchi-facing, Uzbek, sentence-case | **PASS** | "tinglovchi/siz", 3 integration tasks, deadline 29-iyun, reflection, submission |
| Scope correctness (m01–m05 only) | **PASS** | brief + check exclude m05b/m06; note points to P6 (m06) and w3 (m07/m08) |
| No GPU / VRAM | **PASS** | CPU-only (numpy/sklearn) |
| Offline data | **PASS** | bundled checkpoints only (uz_news_corpus, uz_sentiment_mini, uz_mini.vec); no external download/libs |

**Overall: ALL GATES PASS**

---

## w2_check.py — Local Run Output (40/40)

```
[0-qism] Shartnoma mosligi — m01, m02, m03, m04, m05      (21 ✓)
[1-integratsiya] m01 -> m04 — imlo tuzatish + LSH qidiruv
    Buzuq so'rov: 'yangi telifon' -> tuzatilgan: 'yangi telefon'
    Topilgan hujjatlar (top-3): ['yangi telefon bozorda chiqdi', 'yangi telefon shaharda chiqdi'] …
[2-integratsiya] m01 + m02 — sentiment pipeline (ijobiy/salbiy)
[3-integratsiya] m05 Autocomplete
    complete('yangi', 3) = ['dastur', 'loyiha', 'telefon']
    perplexity('yangi telefon chiqdi') = 20.29
[m03] PretrainedEmbedder
    most_similar('toshkent', 5): ['nukus', 'namangan', 'samarqand', 'buxoro', 'xiva']
    oov_rate (2 OOV / 4 token) = 0.50
NATIJA: 40 tekshiruvning hammasi O'TDI ✓
```

---

## Notes

- **Self-check paths** are computed relative to the script
  (`course/milestones/w2_check.py` → repo root), using four bundled checkpoint
  files (news corpus, sentiment mini, mini .vec). Runs offline; no external
  data/libs.
- **`fit_dictionary` dependency**: `m04.correct` requires `fit_dictionary(texts)`
  first (builds the unigram LM P(w)); the contract lists `correct`/`index_docs`/
  `retrieve_lsh` but `fit_dictionary` is the implementation's prerequisite for
  `correct`. The check calls both `fit_dictionary` and `index_docs`.
- **m01→m04 continuity** is inherent: `m04` composes `m01` (`TextPreprocessor`)
  internally for shingles and dictionary tokens.
- **most_similar quality**: the offline 50-dim `uz_mini.vec` places cities
  together (toshkent → nukus/namangan/samarqand/buxoro/xiva), a pedagogically
  clean demo of distributional similarity.
- **Deviation from course_map**: none. Scope (m01–m05, one integration) matches
  `id: w2` exactly.

## Pending

- Tinglovchi-side submission (three integration results + reflection) — collected
  by maintainer at deadline (29-iyun), per the brief.
- **P6** (Day 7, m06 CustomWord2Vec) is the next chronological artifact; **w3
  milestone** (1-iyul) integrates m01–m08 (incl. RNN/GRU/LSTM).
