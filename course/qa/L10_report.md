# QA Report: Day 10 Lecture (L10)

**Artifact**: `course/lectures/d10_ner.tex` → `d10_ner.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 10, `lecture_official_no: 10` — "Nomlangan obyektlarni tanib olish (NER)"
**Paired practice**: P10 (Day 11 — m10 NERTagger, Bi-LSTM IOB2 on WikiANN uz)
**Recap target [B]**: L9 (Bi-LSTM for understanding, not generation) + this morning's P9

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0 (after fixing 2 unbraced comma-titles + 1 stray Cyrillic word); `grep "^!"` empty |
| Zero `Overfull \hbox` > 10pt | **PASS** | `grep "Overfull \hbox ([1-9][0-9]"` empty |
| Slide count (L1–L9 parity, 4 full cycles) | **PASS** | 47 frames before `\appendix` (footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | every cycle has a derivation slide |
| [M] Uzbek-language slide (mandatory) | **PASS** | "Qo'shimchalar va kam ma'lumot NER ni qiynaydi" (-da/-dan suffix → entity boundary; WikiANN ~200, low F1 honestly stated) |
| Every formula has `\bunda{}` key | **PASS** | [G1] IOB2, [G2] per-token tag, [G3] Bi-LSTM, [G4] entity F1; appendix S1 (CRF) / S2 (BIOES) |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] all `[fragile]` |
| **Locked [I1] hand_example** | **PASS** | IOB2: "Malika Toshkentda ishlaydi ." → `[B-PER, B-LOC, O, O]` — verbatim from course_map (categorical) |
| Traceability comment → P10 | **PASS** | [I1] + [Q] carry `# Ma'ruza L10 [I1]-slayd`; [Q] shows `tags == ["B-PER","B-LOC","O","O"]` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | 1 stray Cyrillic (`shунга`→`shunga`) caught by pre-compile scan → fixed; final scan clean |
| Preamble byte-identical to d09 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim |
| Visual review (rendered PNG @95dpi) | **PASS** | locked [I1] tagging table, per-token [I2], Bi-LSTM concat [I3], entity-F1 [I4], [N] schemes table, [Q] bridge, code slides — no overflow/clipping/collisions |
| Aux/PNG auto-cleaned | **PASS** | `.aux/.log/.nav/.out/.snm/.toc/.vrb` + render PNG dir removed; only `.tex`+`.pdf` remain |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Results (→ P10 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I1]** | IOB2 tags for "Malika Toshkentda ishlaydi ." | **[B-PER, B-LOC, O, O]** | **P10 first assert** (course_map lock, categorical) |
| [J1] | "Akmal Navoiy shahrida yashaydi" | [B-PER, B-LOC, I-LOC, O] | task (multi-token entity → I-) |
| [I2] | per-token tag, softmax `[0,2,1]` over {O,B-PER,B-LOC} | B-PER (0.665) | architecture step |
| [J2] | tag, `[3,1,0]` | O | task |
| [I3] | concat fwd `[0.2,0.8]` + bwd `[0.9,0.1]` | [0.2,0.8,0.9,0.1] (dim 4) | Bi-LSTM context |
| [J3] | Bi-LSTM hidden=64 concat dim | 128 | task |
| [I4] | entity P/R/F1 (type error on Toshkent) | 0.5 / 0.5 / 0.5 | entity-level eval |
| [J4] | all-correct entity F1 | 1.0 | task |

The §[I1] cell tags the sentence by hand — **P10's first assert**, matching course_map
Day 10 `hand_example`.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L9 (Bi-LSTM home is tagging); problem = dictionary/suffix |
| Teglash | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | NER task + IOB2 + **locked [I1]** |
| Arxitektura | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | LSTM per-token tagging |
| Bi-LSTM | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | bidirectional context (why NER, not generation) |
| Baholash | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | entity-level P/R/F1 (seqeval) |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1 CRF][S2 BIOES] |

---

## Content Continuity

- **[B] recap** bridges L9 (Bi-LSTM can't generate — needs future) → its proper home:
  sequence tagging (NER), where the whole sentence is known.
- **[E] problem-first** shows dictionary-lookup and suffix-boundary failures *before*
  naming IOB2 / Bi-LSTM.
- **[M] Uzbek**: place-name + suffix ("Toshkentda", "Samarqanddan") boundary issue;
  IOB2 keeps the whole token `B-LOC`; low-resource WikiANN (~200) → low F1, stated honestly.
- **[O] seminal**: Lample et al. (2016) Bi-LSTM-CRF + discussion (low-resource Uzbek NER).
- **[Q] bridge**: TikZ pipeline WikiANN uz → Embedding → Bi-LSTM → m10 NERTagger;
  first assert = the locked [I1] tag sequence.

---

## Compile Fixes Applied

1. **Cyrillic** (pre-compile scan): stray `shунга` (4 Cyrillic letters) in [G3] `\bunda` →
   `shunga`. The recurring Cyrillic-lookalike trap; caught proactively this time.
2. **Unbraced comma-titles** (pgfkeys errors): [G4] `title=...precision, recall, F1` and
   [L4] `title=Yuqori accuracy, yomon NER` → wrapped in `title={...}` (pitfall #1).

Recompiled clean: 0 errors, 0 Overfull, 0 Cyrillic.

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (NER task + schemes; LSTM architecture; Bi-LSTM context;
evaluation), seminal paper, uzbek_angle, locked hand_example (IOB2 tags), and
`gpu_required: true` all match Day 10 `lecture_official_no: 10`.

---

## Pending

- **Overleaf human review** of prose quality — per `uzbek-course-style` (human gate).
- **Next chronological artifact**: **w3 milestone** (1-iyul, m01–m08) — comes *before*
  P10 (Day 11). Then P10 (m10 NERTagger) consumes [I1] as its first assert.
- Commit pushed to origin/rtm (per task instruction).
