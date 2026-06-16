# QA Report: Day 11 Practice (P10)

**Artifacts**: `course/practices/d11_p10_ner.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m10_ner_tagger.py`
**Bundled data**: `course/practices/d11_checkpoints/uz_ner_mini.txt` (original IOB2, 35 sentences)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L10 — Nomlangan obyektlarni tanib olish (NER)
**Next**: L11 (Day 11 lecture — attention / seq2seq)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `json.load` OK; nbformat 4.5; 28 cells (13 code, 15 md); all `id` |
| JSON valid — solutions | **PASS** | `json.load` OK; nbformat 4.5; 28 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True, CPU)** | **PASS (local)** | SOLUTIONS exec'd in sequence on Python 3.13.14 — **10/10 asserts passed**, zero exceptions. **Both paths verified**: torch 2.10+cpu (real `nn.LSTM(bidirectional=True)`) AND forced-reservoir fallback (Bi-LSTM forward + LogReg). |
| Student stub cells compile | **PASS** | All 13 code cells `compile()` clean |
| **Locked assert (gold IOB2)** | **PASS** | §4A gold tags "Malika Toshkentda ishlaydi ." → `[B-PER, B-LOC, O, O]` (pure-python, hand reference — NOT model output) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L10 [I1]-slayd bilan solishtiring` (§4A) |
| Every blanked region has paired assert | **PASS** | 3 blanked cells (§4B fit/predict, §4C entities, §4D F1+save/load) → paired asserts |
| m10 contract conformance | **PASS** | `fit(tagged_sentences)` / `predict->list[tuple[str,str]]` / `entities->list[dict]` / `save` / `load` exact (contracts.py) |
| predict / entities structural | **PASS** | predict = list of (token, tag); tags ∈ IOB2 set; entities = list of dicts with `text`/`label`/`start`/`end` |
| Capstone continuity | **PASS** | real pipeline module (`consumed_by: [15, 16]`); save/load works (m15 agent, Day 16) |
| No GPU / VRAM | **PASS** | CPU-only; small Bi-LSTM; VRAM peak 0 GB |
| Data size | **PASS** | bundled IOB2 corpus ≈ 2 KB (≪ 500 MB; real WikiANN uz online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, reservoir `RandomState(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads IOB2 corpus) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` (corpus "oqituvchi" → "shifokor" to avoid ambiguity) |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | all four artifacts verified clean |

**Overall: ALL GATES PASS**

---

## torch present locally — both paths verified

Local env has **torch 2.10.0+cpu**, so the real `nn.LSTM(bidirectional=True)` NER path was
verified end-to-end on CPU; the torchless reservoir fallback (forced `HAS_TORCH=False`)
was **also** verified.

| Path | predict("Malika Toshkentda ishlaydi .") | train token acc |
|---|---|---|
| torch Bi-LSTM + Softmax | [B-PER, B-LOC, O, O] ✓ | 0.98 |
| reservoir Bi-LSTM + LogReg (forced) | [B-PER, B-LOC, O, O] ✓ | 0.82 |

The locked §4A gold-tag assert is pure-python (hand reference) and path-independent.
`entities("Akmal Samarqandda yashaydi .")` → `[{'text':'Akmal','label':'PER',...}, {'text':'Samarqandda','label':'LOC',...}]` on both paths.

---

## Locked / Verified Results

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I1]** | gold IOB2 tags for "Malika Toshkentda ishlaydi ." | **[B-PER, B-LOC, O, O]** | **P10 first assert** (course_map lock; hand gold, not prediction) |

The §4A cell reproduces the lecture's hand IOB2 tagging — **P10's first assert** — matching
course_map Day 11 paired-lecture L10 `hand_example`. Per the design, this asserts the
**gold** tagging (traceability), not the model's prediction (which is unreliable on ~35
training sentences — honestly noted).

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L10 [C], timing |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, HAS_TORCH, HAS_SEQEVAL |
| §2 Yaxlit natija | 2 | MD+Code | OK — finished NER + entities() demo |
| §3 PRIMM periferiya | 4 | Mixed | OK — CoNLL IOB2 load + tag/index encoding; padding/masking; Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked gold) + 3 blanked (4B fit/predict, 4C entities, 4D F1+save/load) each with paired assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m10 import/contract test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — -da/-dan suffix; why low F1 on small data, exit ticket |

Total: 28 cells (13 code, 15 markdown). Blanked core cells: 3, each paired with an assert.

---

## torch-Optional + seqeval-Optional Design

`torch`/`seqeval` may be absent. m10 and the notebook branch on `HAS_TORCH`/`HAS_SEQEVAL`:
- **Kaggle path**: `nn.Embedding + nn.LSTM(bidirectional=True) + nn.Linear` (per-token tag),
  `CrossEntropyLoss(ignore_index=-100)` + `Adam`; entity-level F1 via seqeval.
- **Offline path** (forced/torchless): random-init Bi-LSTM **forward** → per-token
  `[E[w]; h_fwd; h_bwd]` features → sklearn `LogisticRegression` readout. Token-level
  macro-F1 (sklearn). No fragile BPTT.

Result: runs end-to-end with or without torch/seqeval, CPU-only, no WikiANN download.

---

## Module Conformance (contracts.py)

```
m10 NERTagger (consumed_by: [15, 16]):                              provides:
  fit(tagged_sentences: list[list[tuple[str,str]]]) -> None          ✓
  predict(text: str) -> list[tuple[str, str]]                        ✓  [(token, IOB2-tag), ...]
  entities(text: str) -> list[dict]                                  ✓  [{'text','label','start','end'}, ...]
  save(path) / load(path)                                            ✓  (m15 agent + Day 16 pipeline)
```

---

## Deviation from course_map.yaml

course_map Day 11 `corpus_subset: uz_ner_wikiann` (online, ~200, low F1 stated) and
`gpu_required: true`. The **OFFLINE_FALLBACK** uses a small **original** IOB2 corpus
(`uz_ner_mini.txt`, 35 sentences; WikiANN not bundled — license/download). Local run is
**CPU-only** (GPU is an accelerator). F1 is intentionally low (tiny data) — stated honestly
per the course_map note. CRF layer omitted (Softmax sufficient pedagogically; CRF noted in
L10 appendix).

---

## Pending

- Full Kaggle kernel run with real WikiANN uz on GPU (torch Bi-LSTM, seqeval entity-F1) —
  confirmed when notebooks are published as a Kaggle Dataset (Day 11, 2-iyul-2026).
- **L11** (Day 11 lecture — attention / seq2seq) is the next chronological artifact.
