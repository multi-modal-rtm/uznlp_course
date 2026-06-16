# QA Report: Day 8 Lecture (L8)

**Artifact**: `course/lectures/d08_gru_lstm.tex` → `d08_gru_lstm.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 8, `lecture_official_no: 8` — "Ilg'or RNN arxitekturalari: GRU va LSTM"
**Paired practice**: P8 (Day 9 — m08 GRULSTMClassifier, PyTorch `nn.LSTM`/`nn.GRU` compare)
**Recap target [B]**: L7 (RNN vanishing gradient) + this morning's P7

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0; `grep "^!"` empty on both passes |
| Zero `Overfull \hbox` > 10pt | **PASS** | `grep "Overfull \hbox ([1-9][0-9]"` empty |
| Slide count (L1–L7 parity, 4 full cycles) | **PASS** | 47 frames before `\appendix` (40 content + 6 TOC + title; footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | every cycle has a derivation slide (planned from the start) |
| [M] Uzbek-language slide (mandatory) | **PASS** | "Cell state uzun qo'shimcha zanjirini ushlay oladimi?" (morpheme chain + SOV; LSTM advantage vs small-corpus overfit) |
| Every formula has `\bunda{}` key | **PASS** | [G1] GRU, [G2] cell-state gradient, [G3] LSTM, [G4] param count; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] all `[fragile]` |
| **Locked [I3] hand_example** | **PASS** | forget gate `f_t = σ(1.5) = 0.818` (σ(1.5)=0.8176) — verbatim from course_map |
| Traceability comment → P8 | **PASS** | [I3] + [Q] carry `# Ma'ruza L8 [I3]-slayd`; [Q] shows `abs(f_t - 0.818) < 1e-3` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | all four verified clean (Cyrillic-lookalike trap explicitly checked → 0) |
| Preamble byte-identical to d07 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim; only title/date/`\textnumero` changed |
| Visual review (rendered PNG @95dpi) | **PASS** | GRU [G1] + LSTM [G3] multi-eq aligns, locked [I3], gradient [I2], [N] 3-row table, [Q] split-bridge TikZ — no overflow, no clipping, no collisions |
| Aux/PNG auto-cleaned | **PASS** | `.aux/.log/.nav/.out/.snm/.toc/.vrb` + render PNG dir removed; only `.tex`+`.pdf` remain |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Numbers (→ P8 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I3]** | LSTM forget gate `σ(W_f·0.5 + W_f·1.0 + 0)` | **0.818** | **P8 first assert** (course_map lock) |
| [J3] | cell update `0.818·1.0 + 0.5·0.6` | 1.118 | task (additive cell state) |
| [I1] | GRU update `σ(1.0)`, then gated mix | z=0.731 → h=0.639 | GRU interpolation |
| [J1] | GRU with `z=σ(0)=0.5` | h=0.5 | task |
| [I2] | gradient over 10 steps: RNN `0.42^10` vs LSTM `0.9^10` | 1.7e-4 vs 0.349 | vanishing-gradient contrast |
| [J2] | `0.9^5` | 0.59 | task |
| [I4] | param count d=2,H=2: LSTM `4·10` / GRU `3·10` | 40 / 30 | GRU ≈ 0.75× LSTM |
| [J4] | param count d=3,H=2 | 48 / 36 | task |

All [I]/[J] numbers are mentally/`σ`-table computable; only the forget gate `0.818` is locked.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L7 vanishing gradient; problem = carry info unchanged |
| GRU | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | update/reset gates, interpolation |
| Cell state | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | vanishing gradient → additive cell highway |
| LSTM | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | forget/input/output gates + **locked [I3]** |
| Taqqoslash | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | GRU vs LSTM param count, when to use |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1][S2] |

---

## Content Continuity

- **[B] recap** bridges L7 (why RNN gradient vanishes: `∏(1-h²)W_h`) + P7 (RNN classifier
  struggles on long/SOV sentences) → motivates gated memory.
- **[E] problem-first** shows plain RNN overwrites/squashes state each step *before*
  naming GRU/LSTM.
- **[M] Uzbek**: long suffix chain ('o'rgan-il-gan-lar-dan' ≈ 4 morphemes), SOV
  predicate-at-end → cell state holds meaning; caveat: small Uzbek corpora favor GRU
  (fewer params).
- **[O] seminal**: Hochreiter & Schmidhuber (1997) + discussion (GRU vs LSTM on small
  Uzbek corpus) due before P8.
- **[Q] bridge**: TikZ pipeline nn.Embedding → nn.LSTM/nn.GRU → m08 compare_report;
  first assert = the locked [I3] value.

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (GRU; vanishing→LSTM; LSTM components; GRU-vs-LSTM comparison),
seminal paper, uzbek_angle, locked hand_example (`f_t = 0.818`), and `gpu_required: false`
all match Day 8 `lecture_official_no: 8`.

---

## Pending

- **Overleaf human review** of prose quality (sentence-case, natural Uzbek) — per
  `uzbek-course-style` this is a human gate, not automated.
- **P8** (Day 9 — m08 GRULSTMClassifier) consumes [I3] as its first assert; uses
  PyTorch (torch present locally per P7 — torch-optional design recommended).
- Commit pushed to origin/rtm together with P7's 3 commits (per task instruction).
