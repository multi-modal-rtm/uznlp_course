# QA Report: Day 6 Practice (P5)

**Artifacts**: `course/practices/d06_p5_autocomplete_pos.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone modules**: `capstone/modules/m05_autocomplete.py` + `capstone/modules/m05b_pos_tagger.py`
**Bundled data**: `course/practices/d06_checkpoints/` (uz_lm_corpus.txt, hmm_transition.csv, hmm_emission.csv)
**Date**: 2026-06-15
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L5 — Ehtimollik modellari (N-gram, perplexity, POS/HMM/Viterbi)
**Next**: L6 (Day 6 lecture) → w2 milestone

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `JSON.parse` OK; nbformat 4.5; 37 cells (20 code, 17 md); all `id` |
| JSON valid — solutions | **PASS** | `JSON.parse` OK; nbformat 4.5; 37 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True)** | **PASS (local)** | Solutions executed cell-by-cell on Python 3.13.14 — **every assert passed**, zero exceptions. Pure-python n-gram (nltk not required). |
| Student stub cells compile | **PASS** | All 20 code cells `compile()` clean |
| **Locked assert (Viterbi)** | **PASS** | §4C `δ(VB,t=2)=0.3402` and `tag_sequence==["NN","VB"]` |
| Secondary assert (bigram) | **PASS** | §4A `P(kitob\|men)=2/3` |
| Traceability comments cite lecture | **PASS** | `# Ma'ruza L5 [I4]-slayd` (§4C), `# Ma'ruza L5 [I1]-slayd` (§4A) |
| Every blanked region has paired assert | **PASS** | 4 blanked cells (§4B autocomplete, §4D δ(NN), §4E perplexity, §4F mustaqil) → 4 paired asserts |
| m05 contract conformance | **PASS** | `train(texts,n)`/`complete(prefix,k)->list[str]`/`perplexity(text)->float` exact (+save/load) |
| m05b contract conformance | **PASS** | `train(tagged_sents)`/`tag(tokens)->list[tuple[str,str]]` exact |
| Capstone continuity (uses m01) | **PASS** | both modules use `TextPreprocessor._normalize` (m01) for normalization |
| No GPU / VRAM | **PASS** | CPU-only; pure-python; VRAM peak 0 GB |
| Data size | **PASS** | corpus ≈ 4 KB + 2 small CSV (≪ 500 MB; real uz_news_full online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)` |
| Checkpoint cells present | **PASS** | Checkpoint A (§3, reloads corpus + HMM) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` across all artifacts |

**Overall: ALL GATES PASS**

---

## Locked Numeric Values — P5 Core Asserts

| Assert | Slide | Expected | Computed |
|--------|-------|----------|----------|
| Viterbi `δ(VB, t=2)` (§4C) | L5 [I4] | 0.3402 | 0.3402 |
| Tag sequence (§4C) | L5 [I4] | `[NN, VB]` | `[NN, VB]` |
| Bigram `P(kitob\|men)` (§4A) | L5 [I1] | 2/3 | 0.6667 |
| `δ(NN, t=2)` (§4D) | L5 [J4] | 0.0252 | 0.0252 |

HMM params (from `hmm_transition.csv` / `hmm_emission.csv`) reproduce L5 [I4] exactly:
π(NN)=0.7, π(VB)=0.3; A(NN→VB)=0.6, A(NN→NN)=0.4, A(VB→NN)=0.3, A(VB→VB)=0.7;
B(nlp|NN)=0.9, B(nlp|VB)=0.1, B(yozdi|NN)=0.1, B(yozdi|VB)=0.9.
The §4C Viterbi (linear-space, matching the lecture hand calc) is the traceability vehicle.

---

## Built Results (offline, seed 42)

| Operation | Result |
|---|---|
| `complete("men", 3)` (raw bigram count) | [xat, choy, kitob] |
| `Autocomplete.complete("men", 3)` (Laplace) | [choy, dastur, film] |
| `perplexity("men kitob o'qidim")` | 7.79 (vocab 17) |
| Viterbi `tag(["nlp","yozdi"])` | [NN, VB] (δ(VB,t=2)=0.3402) |
| `POSTagger.tag(["nlp","yozdi"])` (trained) | [(nlp,NN), (yozdi,VB)] |

> Note: §4B uses raw bigram counts; m05 uses Laplace-smoothed probabilities (different
> tie-breaking), so their top-3 differ slightly — both valid rankings. Asserts check
> structural validity (candidates are real 'men'-followers), not an exact ordering.

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK |
| §1 Muhit | 1 | Code | OK — seeds, OFFLINE_FALLBACK, m01 path, HAS_NLTK |
| §2 Yaxlit natija | 2 | MD+Code | OK — inline autocomplete + Viterbi demo |
| §3A PRIMM: n-gram | 3 | MD+Code+MD | OK — nltk (guarded) + pure-python `ngrams_py` |
| §3B PRIMM: HMM CSV | 3 | MD+Code+MD | OK — load π/A/B from CSV |
| Checkpoint A | 1 | Code | OK |
| §4 core | 14 | Mixed | OK — 2 Namuna (4A bigram, 4C Viterbi-locked) + 4 Birgalikda blanks (4B,4D,4E,4F) |
| §5 Loyihaga ulash | 5 | MD+Code | OK — write m05 + m05b, import test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — SOV/agglutination, exit ticket |

Total: 37 cells (20 code, 17 markdown). Blanked core cells: 4, each paired with an assert.

---

## Module Conformance (contracts.py)

```
m05 Autocomplete:                              provides:
  train(texts: list[list[str]], n=2) -> None   ✓
  complete(prefix: str, k=3) -> list[str]       ✓
  perplexity(text: str) -> float                ✓  (+ save/load helpers)

m05b POSTagger:                                provides:
  train(tagged_sentences) -> None               ✓  (estimates π,A,B)
  tag(tokens: list[str]) -> list[tuple[str,str]] ✓  (log-space Viterbi)
```
Both import m01 `TextPreprocessor` and use its `_normalize` (apostrophe + lowercase).
**m05** keeps stop-words (language modeling needs function words — it does NOT call
`m01.preprocess`, which removes stop-words/stems). **m05b** is pedagogical (`consumed_by [] `);
m05 is `consumed_by [16]`.

---

## nltk-Optional Design

`nltk` may be absent; the notebook uses a pure-python `ngrams_py` (zip/slicing) for
n-grams, with `HAS_NLTK` only switching the informational message. m05's n-gram counting
and m05b's Viterbi are pure-python. Result: the notebook runs end-to-end **without nltk
and without the real uz_news_full** — mirroring the gensim/datasketch-optional pattern of
P3/P4. (Locally `nltk` happened to be present, but the code path does not depend on it.)

---

## Offline Data Note

`d06_checkpoints/uz_lm_corpus.txt` — 150 short Uzbek sentences (controlled vocabulary,
function words kept) for the bigram/autocomplete demo. `hmm_transition.csv` /
`hmm_emission.csv` — the L5 [I4] HMM parameters (NN/VB; nlp/yozdi), loaded by §3B and used
by the §4C locked Viterbi. The L5 [I1] bigram assert (P(kitob|men)=2/3) uses a fixed inline
3-sentence corpus (matching the lecture). All offline; no external data/libs.

---

## Deviation from course_map.yaml

None. course_map Day 6 `corpus_subset: uz_news_full` respected (online); bundled corpus is
the documented OFFLINE_FALLBACK. Both capstone modules (m05 + m05b) built per `capstone_module`
and `capstone_module_b`. License `uz_news_full` = **CONFIRMED** in `LICENSES.md`.

---

## Pending

- Full Kaggle kernel run with real `uz_news_full` (and optional nltk) — confirmed when
  notebooks are published as a Kaggle Dataset (Day 6, 23-iyun-2026).
- **L6** (Day 6 lecture — Word2Vec CBOW) and **w2 milestone** (24-iyun, m01–m05) — next.
