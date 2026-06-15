# QA Report: Day 7 Practice (P6)

**Artifacts**: `course/practices/d07_p6_word2vec.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m06_custom_word2vec.py`
**Bundled data**: `course/practices/d07_checkpoints/uz_w2v_corpus.txt` (98 sentences)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L6 — Word2Vec (CBOW), neyron embeddinglar
**Next**: L7 (Day 7 lecture — RNN)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `json.load` OK; nbformat 4.5; 28 cells (13 code, 15 md); all `id` |
| JSON valid — solutions | **PASS** | `json.load` OK; nbformat 4.5; 28 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True)** | **PASS (local)** | All SOLUTIONS code cells exec'd in sequence on Python 3.13.14, **gensim absent** — **8/8 asserts passed**, zero exceptions. Pure-numpy CBOW. |
| Student stub cells compile | **PASS** | All 13 code cells `compile()` clean (0 SyntaxError) |
| **Locked assert (CBOW projection)** | **PASS** | §4A `cbow_input = mean([0.5,0.3],[0.1,0.7]) = [0.3,0.5]` |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L6 [I2]-slayd bilan solishtiring` (§4A) |
| Every blanked region has paired assert | **PASS** | 3 blanked cells (§4B train, §4C most_similar, §4D save/load+OOV) → each paired with a self-check assert |
| m06 contract conformance | **PASS** | `train(texts, vector_size, window, min_count, epochs)` / `embed`/`most_similar`/`save`/`load` exact (contracts.py) |
| Capstone continuity (uses m01) | **PASS** | corpus cleaned via `TextPreprocessor.preprocess` (m01) → `list[list[str]]` (gensim `LineSentence` equivalent) |
| No GPU / VRAM | **PASS** | CPU-only; pure-numpy CBOW; VRAM peak 0 GB |
| Data size | **PASS** | 98-sentence corpus ≈ 3 KB (≪ 500 MB; real uz_news_full online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`; numpy CBOW `RandomState(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads cleaned sentences) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` across all artifacts (corpus line "talabalar" fixed → "bolalar") |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | all four artifacts verified clean |

**Overall: ALL GATES PASS**

---

## Locked / Verified Numbers

| Assert | Slide | Expected | Computed |
|--------|-------|----------|----------|
| §4A CBOW projection `cbow_input` | L6 [I2] | `[0.3, 0.5]` | `[0.3, 0.5]` |

The §4A cell reproduces the lecture's hand example (mean of context embeddings) and
is the traceability vehicle → **this is P6's first assert**, matching course_map
Day 7 `hand_example: CBOW_input = [0.3, 0.5]`.

---

## Local CBOW Quality (offline, pure-numpy, seed 42)

`most_similar` on the bundled corpus produces coherent semantic clusters — the
distributional hypothesis in action (words sharing context get near vectors):

| Query | Top-3 most_similar | Cluster |
|---|---|---|
| `toshkent` | samarqand (0.99), buxoro (0.99), namangan (0.96) | cities |
| `telefon` | kompyuter (1.0), dastur (1.0), dokon (0.96) | tech |
| `osh` | palov (1.0), somsa (0.98), dasturxon (0.96) | food |
| `kitob` | oqish (0.88), maktab (0.83), bilim (0.75) | study |

> The §4C assert checks **structural** validity (list of `(str, float)` with cosine
> in [-1,1], len ≤ k), not exact ordering — but the clusters above show the
> pure-numpy CBOW genuinely learns meaning on a tiny corpus.

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L6 [C], timing |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, m01 path, HAS_GENSIM |
| §2 Yaxlit natija | 2 | MD+Code | OK — finished CBOW + most_similar demo |
| §3 PRIMM periferiya | 5 | Mixed | OK — m01 cleaning (LineSentence), TensorBoard metadata; Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked) + 3 blanked (4B train, 4C most_similar, 4D save/load) each with paired assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m06 import/contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — agglutination/vocab explosion, exit ticket |

Total: 28 cells (13 code, 15 markdown). Blanked core cells: 3, each paired with an assert.

---

## gensim-Optional Design (m03 pattern)

`gensim` may be absent. m06 and the notebook branch on `HAS_GENSIM`:
- **Kaggle path**: `gensim.models.Word2Vec(..., sg=0)` (CBOW), `.kv` save,
  `wv.most_similar` — shown as the production path.
- **Offline path** (local, no gensim): pure-numpy CBOW with negative sampling —
  projection = context mean (L6 [I2]); `embed`/`most_similar` via cosine; pickle
  save/load.

Result: the notebook runs end-to-end **without gensim and without uz_news_full** —
mirroring the gensim/datasketch/nltk-optional pattern of P3/P4/P5. (`save`/`load`
use a `.kv`-named file; offline both ends are pickle, fully consistent.)

---

## Module Conformance (contracts.py)

```
m06 CustomWord2Vec:                                              provides:
  train(texts, vector_size=100, window=5, min_count=3, epochs=10) -> None  ✓
  embed(word: str) -> np.ndarray   (OOV -> zero-vector)                     ✓
  most_similar(word: str, n=5) -> list[tuple[str, float]]                   ✓
  save(path: str) / load(path: str)                                        ✓
```
Uses m01 `TextPreprocessor` for corpus cleaning. `consumed_by: [8, 9]`
(m08 pretrained Embedding init, m09 generator).

---

## Deviation from course_map.yaml

None. course_map Day 7 `corpus_subset: uz_news_full` respected (online); bundled
`uz_w2v_corpus.txt` is the documented OFFLINE_FALLBACK. CBOW (`sg=0`),
`vector_size`/`window`/`min_count` per `core`. License `uz_news_full` =
**CONFIRMED** in `LICENSES.md`. Notebook demo uses `vector_size=50, epochs=30`
(CPU/tiny-corpus); full-scale `vector_size=100, epochs=10` noted in a comment.

---

## Pending

- Full Kaggle kernel run with real `uz_news_full` (and gensim CBOW + TensorBoard
  projector) — confirmed when notebooks are published as a Kaggle Dataset
  (Day 7, 25-iyun-2026).
- **L7** (Day 7 lecture — RNN) is the next chronological artifact.
