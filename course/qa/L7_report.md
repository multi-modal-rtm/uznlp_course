# QA Report: Day 7 Lecture (L7)

**Artifact**: `course/lectures/d07_rnn.tex` → `d07_rnn.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 7, `lecture_official_no: 7` — "Rekurrent neyron tarmoqlar (RNN)"
**Paired practice**: P7 (Day 8 — m07 RNNClassifier, PyTorch `nn.RNN` text classification)
**Recap target [B]**: L5 (N-gram fixed window) + L6 (embeddings) + this morning's P6

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0; `grep "^!"` empty (after fixing math-mode in tcolorbox titles) |
| Zero `Overfull \hbox` > 10pt | **PASS** | `grep "Overfull \hbox ([1-9][0-9]"` empty |
| Slide count (L1–L6 parity, 4 full cycles) | **PASS** | 47 frames before `\appendix` (40 content + 6 TOC + title; footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | every cycle has a derivation slide (the L6 [H2] gap explicitly avoided) |
| [M] Uzbek-language slide (mandatory) | **PASS** | "SOV tartibi va uzun qo'shimchalar RNN ni sinaydi" (kesim oxirida + morpheme chain → vanishing gradient → LSTM/GRU) |
| Every formula has `\bunda{}` key | **PASS** | [G1] feedforward LM, [G2] RNN recurrence, [G3] BPTT, [G4] output head; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] all `[fragile]` |
| **Locked [I2] hand_example** | **PASS** | `h_1 = tanh([1,0]) = [0.762, 0.000]` (tanh(1)≈0.7616, tanh(0)=0) — verbatim from course_map |
| Traceability comment → P7 | **PASS** | [I2] + [Q] carry `# Ma'ruza L7 [I2]-slayd`; [Q] shows `np.allclose(h1, [0.762, 0.0], atol=1e-3)` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | all four verified clean |
| Preamble byte-identical to d06 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim; only title/date/`\textnumero` changed |
| Visual review (rendered PNG @95dpi) | **PASS** | unrolling TikZ, locked [I2], BPTT [H3], 4-col [N] table, PyTorch [K4], [Q] bridge — no overflow, no clipping, no collisions |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Numbers (→ P7 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I2]** | `h_1 = tanh(W_h·[0,0] + W_x·[1,0])` | **[0.762, 0.000]** | **P7 first assert** (course_map lock) |
| [J2] | `h_1` with `x_1=[0,1]` | [0.000, 0.762] | task (same shape, diff numbers) |
| [I1] | concat `[e_men; e_kitobni]` | [0.5,0.3,0.2,0.6] (dim 4) | fixed-window input |
| [I3] | `h_2 = tanh([0.762,1])` | [0.642, 0.762] | sequential memory (chained from h_1) |
| [J3] | `1 − 0.762²` (tanh derivative) | 0.419 (<1) | vanishing-gradient intuition |
| [I4] | `softmax([1,0])` from `h_T=[1,0]` | ŷ(ijobiy)=0.731 → ijobiy | many-to-one classification |
| [J4] | `softmax([0,1])` from `h_T=[0,1]` | ŷ(ijobiy)=0.269 → salbiy | task |

All [I]/[J] numbers are mentally/`tanh`-table computable; only `h_1` is locked in course_map.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L5+L6+P6; problem = variable length / far predicate |
| Neyron LM | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | fixed-window feedforward LM, its limits |
| RNN | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | recurrence + **locked [I2]** |
| Yoyilish | Cycle 3: [F3]+unroll-TikZ [G3][H3][I3][J3][K3][L3] | unrolling, BPTT, vanishing gradient |
| Qo'llash | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | LM vs classification; PyTorch `nn.RNN` |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1][S2] |

---

## Content Continuity

- **[B] recap** bridges L5 (N-gram fixed window can't see far predicate) + L6/P6
  (embeddings give word vectors but no sequence memory) → motivates recurrent state.
- **[E] problem-first** shows fixed-window feedforward LM blindness and parameter
  explosion *before* naming RNN.
- **[M] Uzbek**: SOV order (predicate last) demands long memory; long suffix chains
  ('o'rgan-il-gan-lar-dan'); vanishing gradient → pointer to LSTM/GRU (L8).
- **[O] seminal**: Rumelhart, Hinton, Williams (1986) backprop + discussion question
  (how does RNN learn long dependency if gradient vanishes?) due before P7.
- **[Q] bridge**: TikZ pipeline m01 → nn.Embedding → nn.RNN → m07 RNNClassifier;
  first assert = the locked [I2] value.

---

## Compile Fix Applied

Initial compile raised math-mode errors: three tcolorbox titles ([I2], [I3], [I4])
contained bare subscripts (`h_0`, `W_h`, `x_1`, `h_T`) in text mode. Fixed by wrapping
the variable expressions in `$...$` inside braced titles. (Recurring LaTeX pitfall #2 —
bare `_` in text mode.) `\begin{psmallmatrix}` (mathtools, not loaded) replaced with
`\left(\begin{smallmatrix}...\end{smallmatrix}\right)` (amsmath). Recompiled clean.

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (N-gram→neural LM; hidden state; unrolling/BPTT; LM+classification),
seminal paper, uzbek_angle, locked hand_example (`h_1 = [0.762, 0.000]`), and
`gpu_required: false` all match Day 7 `lecture_official_no: 7`.

---

## Pending

- **Overleaf human review** of prose quality (sentence-case, natural Uzbek) — per
  `uzbek-course-style` this is a human gate, not automated.
- **P7** (Day 8 — m07 RNNClassifier) consumes [I2] as its first assert. Note: P7 uses
  PyTorch; local env has no torch — P7 will need a torch-optional strategy (flagged in
  HOLAT_HISOBOT.md).
- Commit must be pushed to origin/rtm after approval.
