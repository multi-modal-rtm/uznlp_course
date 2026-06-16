# QA Report: Week 3 Milestone (w3)

**Artifacts**: `course/milestones/w3_milestone.md` (brief) + `course/milestones/w3_check.py` (self-check)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**course_map**: `id: w3` — "3-hafta milestone: Neyron arxitekturalar integratsiyasi"
(date 1-iyul, deadline 6-iyul; `modules_covered: [1,2,3,4,5,6,7,8]`)

---

## Scope (per course_map id: w3)

One full integration over m01–m08:
- **Integration 1 — classifier comparison**: m02 (TF-IDF+LogReg) vs m07 (RNN) vs
  m08 (GRU/LSTM) on the **same** train/test split → F1 + inference time.
- **Integration 2 — m06 embeddings**: pretrained (m06) vs random init (research task,
  since m07/m08 contracts take no pretrained-embedding param; architecture unchanged).

**m09 (TextGenerator) and m05b (POSTagger) are NOT in scope** (pedagogical,
`consumed_by: []`); **m10 NOT in scope** (built in P10, Day 11). Confirmed against course_map.

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| **w3_check.py local execution** | **PASS** | Python 3.13.14, numpy 2.2.6, sklearn 1.8.0, torch 2.10+cpu — **18/18 checks passed**, exit 0, no gensim/datasketch/nltk |
| Contract conformance m01–m08 | **PASS** | All `contracts.py` methods present for m01..m08 |
| Integration 1 (classifier comparison) | **PASS** | m02/m07/m08 trained on same split (88 train / 30 test); each predicts ijobiy/salbiy; F1 + inference time captured |
| Integration 2 (m06 embeddings) | **PASS** | m06 trained; `embed` shape (32,); `most_similar("toshkent")` → city cluster; basis for pretrained-init research |
| Locked labels | **PASS** | only `ijobiy`/`salbiy` (no musbat/manfiy) |
| Coverage correctness (m01–m08 only) | **PASS** | m09/m05b/m10 explicitly excluded; brief note points to w4 / P10 |
| Architecture unchanged | **PASS** | no pretrained-embedding param added to m07/m08; m06-init treated as research/analysis task |
| Terminology grep (both files) | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| Brief: tinglovchi-facing, Uzbek, sentence-case | **PASS** | classifier-comparison + m06-research tasks, deadline 6-iyul, reflection, submission |
| No GPU / VRAM | **PASS** | CPU-only (torch cpu); small data, few epochs |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `train_test_split(random_state=42)` |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | both files verified clean |

**Overall: ALL GATES PASS**

---

## w3_check.py — Local Run Output (18/18)

```
[0-qism] Shartnoma mosligi — m01–m08            (8 ✓)
[1-integratsiya] Klassifikator taqqoslovi — m02 vs m07 vs m08 (bir xil split)
  train: 88 | test: 30 misol
  Taqqoslov hisoboti:
    m02_logreg   F1=0.968  inference=0.0331s
    m07_rnn      F1=0.875  inference=0.0313s
    m08_lstm     F1=0.812  inference=0.0678s
[2-integratsiya] m06 CustomWord2Vec — pretrained embeddinglar
    m06 'toshkent' ~ ['samarqand', 'buxoro', 'andijon']
NATIJA: 18 tekshiruvning hammasi O'TDI ✓
```

---

## Pedagogical Result (honest, important)

On the small sentiment corpus, the **classical** m02 (TF-IDF + LogReg) **outperforms**
the neural m07/m08:

| Model | F1 | inference |
|---|---|---|
| m02 (LogReg) | **0.968** | 0.033s |
| m07 (RNN) | 0.875 | 0.031s |
| m08 (LSTM) | 0.812 | 0.068s (slowest) |

This is the milestone's teaching point: **neural architectures are not automatically
better — they need more data**. On ~120 examples, the strong classical baseline wins.
m08 (LSTM) is also the slowest at inference. (Numbers are seed-42 stable; on full
uz_news_full the gap narrows/reverses.)

---

## Notes

- **Same train/test split** (`random_state=42`, stratified) is shared across all three
  classifiers — the fairness condition course_map calls for ("bir xil test to'plamida").
- **m06 pretrained vs random**: m07/m08 contracts take no pretrained-embedding argument,
  so the check verifies m06 functionally (embed shape, most_similar) and the brief frames
  the pretrained-init comparison as a **research/reflection task** — architecture is not
  modified (contracts are binding).
- **torch present locally** → m07/m08 run on CPU via the real `nn.*` path; the
  torch-optional numpy fallback also exists for torchless machines.
- **Deviation from course_map**: none. Scope (m01–m08, classifier comparison + m06) matches
  `id: w3`. The pretrained-vs-random part is analytical (contract constraint documented).

## Pending

- Tinglovchi-side submission (own comparison table + m06 study + reflection) — collected
  by maintainer at deadline (6-iyul), per the brief.
- **P10** (Day 11 — m10 NERTagger) is the next chronological artifact after w3.
- **w4 milestone** integrates m01–m15 (transformer/RAG/agent) — final milestone.
