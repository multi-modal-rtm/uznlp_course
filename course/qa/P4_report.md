# QA Report: Day 5 Practice (P4)

**Artifact**: `course/practices/d05_p4_spell_lsh.ipynb` + `d05_p4_spell_lsh_SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m04_spell_lsh_retriever.py`
**Bundled data**: `course/practices/d05_checkpoints/uz_news_corpus.txt`
**Date**: 2026-06-15
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L4 — Masofaga asoslangan qidiruv va imlo tuzatish
**Next**: w1 milestone (m01–m04 integratsiyasi)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `JSON.parse` OK; nbformat 4.5; 34 cells (18 code, 16 md); all `id` |
| JSON valid — solutions | **PASS** | `JSON.parse` OK; nbformat 4.5; 34 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True)** | **PASS (local)** | Solutions executed cell-by-cell on Python 3.13.14 / numpy 2.2.6 — **every assert passed**, zero exceptions. **No datasketch required.** |
| Student stub cells compile | **PASS** | All 18 code cells `compile()` clean |
| Locked assert (edit_distance) | **PASS** | §4A `edit_distance("qo'l","ko'l")==1` and `("dastur","dastir")==1` |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L4 [I3]-slayd bilan solishtiring` (§4A) |
| Every blanked region has paired assert | **PASS** | 4 blanked cells (§4B correct, §4C LSH, §4D speed, §4E mustaqil) → 4 paired asserts |
| m04 contract conformance | **PASS** | `correct/edit_distance/index_docs/retrieve_lsh/save/load` exact; save/load round-trip OK |
| Capstone continuity (uses m01) | **PASS** | §3B + m04 use `TextPreprocessor` (m01) for tokenization/normalization |
| No GPU / VRAM dependency | **PASS** | CPU-only; pure-python DP + numpy MinHash; VRAM peak 0 GB |
| Data size | **PASS** | bundled corpus ≈ 5.7 KB (≪ 500 MB; real uz_news_full ~1000+ docs, online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`; MinHash `RandomState(42)` |
| Checkpoint cells present | **PASS** | Checkpoint A (§3, reloads corpus + dict) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` across all 4 artifacts |

**Overall: ALL GATES PASS**

---

## Locked Numeric Value — P4 First Core Assert

| Assert | Slide | Expected | Computed |
|--------|-------|----------|----------|
| `edit_distance("qo'l","ko'l")` (§4A) | L4 [I3] | 1 | 1 |
| `edit_distance("dastur","dastir")` (§4A) | L4 [I3] | 1 | 1 |

DP (Levenshtein) over the strings; `q→k` / `u→i` single substitution. This is the
L4→P4 traceability link (`# Ma'ruza L4 [I3]-slayd`).

---

## Loaded/Built Results (offline bundled corpus, seed 42)

Deterministic on the bundled `uz_news_corpus.txt` (160 docs, 22-word vocab after m01):

| Operation | Result |
|---|---|
| `correct("telfon")` (noisy channel) | **telefon** (edit-1, high P(w)) |
| `correct("internt")` | **internet** |
| `correct("kompyutr")` | **kompyuter** |
| LSH candidates for a query | 4 of 160 (avg 11.8 across 20 queries) |
| k-NN vs LSH speed | k-NN ≈ 2.7 ms (160 comparisons) · LSH ≈ 1.4 ms (≈12 candidates) |
| `retrieve_lsh(corpus[0])[0]` | == corpus[0] (matches k-NN top) |
| §6 apostrophe edit distances | qo'l/qol=1, to'g'ri/togri=2, ma'no/mano=1 |

> Note: on this small corpus the wall-clock gap is small; the deterministic,
> assertable signal is **candidate count ≪ N** (LSH prunes the search). The notebook
> states LSH's advantage grows with corpus size.

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header + timings | 1 | MD | OK |
| §1 Muhit tekshiruvi | 1 | Code | OK — seeds, OFFLINE_FALLBACK, m01 path, HAS_DATASKETCH |
| §2 Yaxlit natija | 2 | MD+Code | OK — inline correct + similar-docs demo |
| §3A PRIMM: MinHash LSH g'oyasi | 3 | MD+Code+MD | OK — datasketch API (commented/guarded) + portable MinHash |
| §3B PRIMM: unigram dict + P(w) | 3 | MD+Code+MD | OK — uses m01 |
| Checkpoint A | 1 | Code | OK |
| §4 core (so'nuvchi tayanch) | 13 | Mixed | OK — 1 Namuna (4A locked) + 4 Birgalikda blanks (4B,4C,4D,4E*) |
| §5 Loyihaga ulash | 5 | MD+Code | OK — write m04, import+contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — apostrophe/agglutination + LSH approximation, exit ticket |

Total: 34 cells (18 code, 16 markdown). Blanked core cells: 4, each paired with an assert.
(\*4E is the Mustaqil — student stub blanked; solution full.)

---

## m04 SpellLSHRetriever — Signature Conformance

```
contracts.py expects:                          m04 provides:
correct(word: str) -> str                      ✓  (noisy channel argmax P(w)·α^edit)
edit_distance(s1, s2) -> int                   ✓  (Levenshtein DP)
index_docs(texts: list[str]) -> None           ✓  (MinHash + LSH banding)
retrieve_lsh(query: str, k=5) -> list[str]     ✓  (candidates → Jaccard top-k)
save(path: str) -> None                        ✓  (pickle state)
load(path: str) -> None                        ✓  (round-trip verified)
```
Helper `fit_dictionary(texts)` added (learns P(w) for `correct`) and
`lsh_candidates(query)` (exposes the pruned candidate set for the speed lesson) —
beyond-contract helpers; all contract methods present with exact signatures.

---

## datasketch-Optional Design

`datasketch` is **not** available locally (and may be absent in some kernels), so
**m04 contains a self-contained pure-python/numpy MinHash + LSH** (stable `crc32`
token hashing → deterministic) — it requires no external LSH library and is fully
testable offline. The notebook **§3A periphery** shows the industry `datasketch.MinHashLSH`
API as commented/guarded reference (the Kaggle path), with `HAS_DATASKETCH` falling
back to the portable implementation. `edit_distance` and `correct` are pure-python.
Result: the notebook runs end-to-end **without datasketch and without the 1000-doc
`uz_news_full`** (240 MB) — mirroring P3's gensim-optional pattern.

---

## Offline Data Note

`d05_checkpoints/uz_news_corpus.txt` is a **small structured sample** (160 short
Uzbek "news"-style sentences, generated from a controlled vocabulary so words repeat
→ LSH clusters form and dictionary frequencies are meaningful) standing in for the
real `uz_news_full` (loaded online). It lets `correct()` (telfon→telefon),
`retrieve_lsh`, and the k-NN-vs-LSH speed comparison all run offline.

---

## Deviation from course_map.yaml

None. course_map Day 5 `corpus_subset: uz_news_full` is respected (online path);
the bundled corpus is the documented OFFLINE_FALLBACK sample. License:
`uz_news_mini/uz_news_full` is **CONFIRMED (educational use)** in `LICENSES.md`.

---

## Pending

- Full Kaggle kernel run with the **real** `uz_news_full` (and optionally `datasketch`)
  — to be confirmed when notebooks are published as a Kaggle Dataset (Day 5, 22-iyun-2026).
- **w1 milestone** (`milestones/w1_milestone.md` + `w1_check.py`) — integrates m01–m04;
  next deliverable for Week 1 closure.
