# QA Report: Day 4 Practice (P3)

**Artifact**: `course/practices/d04_p3_embeddings.ipynb` + `d04_p3_embeddings_SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m03_pretrained_embedder.py`
**Bundled data**: `course/practices/d04_checkpoints/uz_mini.vec`
**Date**: 2026-06-15
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L3 — Vektor fazo modellari va semantik munosabatlar
**Next practice bridge**: P4 (imlo tuzatish + LSH, m04)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `JSON.parse` OK; nbformat 4.5; 34 cells (18 code, 16 md); all cells have `id` |
| JSON valid — solutions | **PASS** | `JSON.parse` OK; nbformat 4.5; 34 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True)** | **PASS (local)** | Solutions notebook executed cell-by-cell on Python 3.13.14 / sklearn 1.8.0 / numpy 2.2.6 — **every assert passed**, zero exceptions. **No gensim required.** |
| Student stub cells compile | **PASS** | All 18 code cells `compile()` clean (blanked stubs valid) |
| Locked assert (cosine = 2/3) | **PASS** | §4A `cosine_similarity([[1,1,1,0]],[[1,1,0,1]]) = 0.6667`; `abs - 0.667 < 1e-3` |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L3 [I2]-slayd ... (cos = 2/3 ≈ 0.667)` (§4A) |
| Every blanked region has paired assert | **PASS** | 4 blanked cells (§4B most_similar, §4C analogy, §4D oov_rate, §4E mustaqil) → 4 paired asserts |
| Solutions fills all blanks | **PASS** | All solution cells execute; no residual `None`/`pass` reaching asserts |
| m03 contract conformance | **PASS** | `load/embed/most_similar/oov_rate` match `contracts.py`; OOV→zero vector; shape (50,) |
| Capstone continuity (uses m01) | **PASS** | §5 import test runs `TextPreprocessor` (m01) → `oov_rate` on m01 tokens |
| No GPU / VRAM dependency | **PASS** | CPU-only; pretrained vectors loaded from disk; VRAM peak 0 GB |
| Data size | **PASS** | bundled `uz_mini.vec` ≈ 16 KB (≪ 500 MB; real cc_uz_100k.kv is ~240 MB, online only) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`; PCA `random_state=42` |
| Checkpoint cells present | **PASS** | Checkpoint A (§3, reloads vectors) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` across all 4 artifacts |

**Overall: ALL GATES PASS**

---

## Locked Numeric Value — P3 First Core Assert

| Assert | Slide | Expected | Computed | Tolerance |
|--------|-------|----------|----------|-----------|
| cosine(a, b) (§4A) | L3 [I2] | 2/3 ≈ 0.667 | 0.6667 | 1e-3 |

Vectors verbatim from L3 [I2] hand example: `a=(1,1,1,0)` ("nlp juda qiziq"),
`b=(1,1,0,1)` ("nlp juda foydali"), vocab (nlp, juda, qiziq, foydali).
`cosine = 2/(√3·√3) = 2/3`. This is the L3→P3 traceability link.

---

## Trained/Loaded-Model Results (offline bundled sample, seed 42)

Deterministic on the bundled `uz_mini.vec` (37 words, 50-dim, structured vectors):

| Operation | Result |
|---|---|
| `most_similar('toshkent', 5)` | nukus, namangan, samarqand, buxoro, xiva — all Uzbek cities (cos ≈ 0.85) |
| Analogy `toshkent − uzbekiston + rossiya` | **moskva** (cos ≈ 0.956) — capital relation |
| `oov_rate` (sample with mashina, banan) | 0.333 (2 of 6 OOV) |
| `oov_rate` of m01 tokens of a sentence | 0.250 |
| PCA `explained_variance_ratio_` | [0.269, 0.206] (≈ 47.5% in 2D) |
| §6 apostrophe variants (to'g'ri / tog'ri / togri) | all OOV — illustrates normalization need |
| §6 agglutinative forms (toshkent + 3 suffixed) | 0.750 OOV |

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header + timings | 1 | MD | OK |
| §1 Muhit tekshiruvi | 1 | Code | OK — seeds, OFFLINE_FALLBACK, m01/m03 path, gensim optional |
| §2 Yaxlit natija | 2 | MD+Code | OK — inline load + most_similar demo |
| §3A PRIMM: yuklash | 3 | MD+Code+MD | OK — gensim `.kv` (commented) + `.vec` offline |
| §3B PRIMM: PCA | 3 | MD+Code+MD | OK — PCA 2D + scatter (matplotlib-safe) |
| Checkpoint A | 1 | Code | OK |
| §4 core (so'nuvchi tayanch) | 13 | Mixed | OK — 1 Namuna (4A locked) + 3 Birgalikda blanks (4B,4C,4D) + 1 Mustaqil (4E) |
| §5 Loyihaga ulash | 5 | MD+Code | OK — write m03, import+contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — OOV/apostrophe study, exit ticket |

Total: 34 cells (18 code, 16 markdown). Blanked core cells: 4, each paired with an assert.

---

## m03 PretrainedEmbedder — Signature Conformance

```
contracts.py expects:                          m03 provides:
load(path: str) -> None                        ✓  (.kv via gensim | .vec via numpy)
embed(word: str) -> np.ndarray                 ✓  (shape (50,) float32; OOV → zeros)
most_similar(word, n=5) -> list[tuple]         ✓  (cosine over normalized matrix)
oov_rate(texts: list[list[str]]) -> float      ✓  ([0,1])
```
`__init__()` holds words/index/raw+normalized matrices. **gensim-optional:** import is
inside `load()` only on the `.kv` path → module imports and runs with numpy alone.

---

## Offline Data Note

`d04_checkpoints/uz_mini.vec` is a **small structured sample** (37 Uzbek words, 50-dim,
word2vec text format) authored for `OFFLINE_FALLBACK=True` execution — the real
`cc_uz_100k.kv` (~240 MB, Common Crawl Uzbek) is loaded only online (Kaggle, via gensim).
Vectors are constructed with concept dimensions (country / city / capital / food / tech)
so that `most_similar` returns sensible neighbours and the capital analogy
(`toshkent − uzbekiston + rossiya = moskva`) holds — letting the notebook run fully
**without gensim and without the 240 MB file**. Format: `n dim` header + `word v1…v50`.

---

## Deviation from course_map.yaml

None. course_map Day 4 `corpus_subset: cc_uz_100k_kv` is respected (online path);
the bundled `.vec` is the documented OFFLINE_FALLBACK sample, mirroring how P1/P2
bundle small offline data. License: `cc_uz_100k_kv` is **CONFIRMED** in `LICENSES.md`
(Common Crawl Uzbek, research/education permitted).

---

## Pending

- Full Kaggle kernel run with the **real** `cc_uz_100k.kv` (gensim) + matplotlib PCA
  scatter — to be confirmed when notebooks are published as a Kaggle Dataset
  (Day 4, 19-iyun-2026).
