# QA Report: Day 11 Lecture (L11)

**Artifact**: `course/lectures/d11_seq2seq_attention.tex` → `d11_seq2seq_attention.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 11, `lecture_official_no: 11` — "Neyron mashina tarjimasi: Seq2Seq va Attention"
**Paired practice**: P11 (Day 12 — m11 Seq2SeqTranslator, LSTM enc-dec + Bahdanau attention, OPUS-100 uz-en)
**Recap target [B]**: L9/L10 (LSTM sequence models, Bi-LSTM) + this morning's P10

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0 (after fixing `\ding{55}` → `$\times$`, pifont not loaded); `grep "^!"` empty |
| Zero `Overfull \hbox` > 10pt | **PASS** | `grep "Overfull \hbox ([1-9][0-9]"` empty |
| Slide count (L1–L10 parity, 4 full cycles) | **PASS** | 47 frames before `\appendix` (footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | every cycle has a derivation slide |
| [M] Uzbek-language slide (mandatory) | **PASS** | "SOV-SVO konversiya va attention alignment" (read→o'qidim alignment; free word order) |
| Every formula has `\bunda{}` key | **PASS** | [G1] Seq2Seq, [G2] context, [G3] attention energy/softmax, [G4] BLEU; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] all `[fragile]` |
| **Locked [I3] hand_example** | **PASS** | attention `α = softmax([2,1,3]) = [0.245, 0.090, 0.665]` — verbatim from course_map |
| Traceability comment → P11 | **PASS** | [I3] + [Q] carry `# Ma'ruza L11 [I3]-slayd`; [Q] shows `np.allclose(alpha, [0.245,0.090,0.665], atol=1e-3)` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | pre-compile Cyrillic scan = 0 (clean from the start this time); final scan confirms |
| Preamble byte-identical to d10 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim |
| Visual review (rendered PNG @95dpi) | **PASS** | locked [I3] attention, [I4] BLEU (✓/× marks), encoder-decoder code, [N] 4-row table, [Q] bridge — no overflow/clipping/collisions |
| Aux/PNG auto-cleaned | **PASS** | `.aux/.log/.nav/.out/.snm/.toc/.vrb` + render PNG dir removed; only `.tex`+`.pdf` remain |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Numbers (→ P11 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I3]** | attention `α = softmax([2,1,3])` | **[0.245, 0.090, 0.665]** | **P11 first assert** (course_map lock) |
| [J3] | context `Σ α_i h_i` (h=[1,0],[0,1],[1,1]) | [0.910, 0.755] | weighted context |
| [I1] | fixed context `c = h_3` | [1,1] | bottleneck (info loss) |
| [J1] | average context | [0.667, 0.667] | task |
| [I2] | weighted context (α=[0.2,0.3,0.5]) | [0.7, 0.8] | attention concept |
| [J2] | weighted context (α=[0.5,0.5,0]) | [0.5, 0.5] | task |
| [I4] | BLEU-1 (2/3 match, BP=1) | 0.667 | translation eval |
| [J4] | BLEU-1 (2/4 match) | 0.5 | task |

All [I]/[J] numbers are mentally/`exp`-table computable; only the attention softmax is locked.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L9/L10; problem = fixed-context bottleneck |
| Enkoder-Dekoder | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | Seq2Seq + information bottleneck |
| Attention | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | per-step weighted context |
| Q/K/V | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | energy→softmax→context + **locked [I3]** |
| BLEU | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | n-gram precision + brevity penalty |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1 additive/dot][S2 full BLEU] |

---

## Content Continuity

- **[B] recap** bridges L9/L10 (LSTM seq models) → encoder-decoder; the bottleneck of a
  single context vector motivates attention.
- **[E] problem-first** shows the fixed-context bottleneck (long-sentence info loss)
  *before* naming attention.
- **[M] Uzbek**: SOV→SVO reordering (Uzbek "o'qidim" last → English "read" early);
  attention alignment links target↔source independent of position; free word order.
- **[O] seminal**: Bahdanau et al. (2015) + discussion (attention for Uzbek free word order).
- **[Q] bridge**: TikZ pipeline OPUS-100 uz-en → LSTM encoder → Bahdanau attention →
  m11 Seq2SeqTranslator; first assert = the locked [I3] attention weights.
- **Cross-lecture tie**: explicitly notes the attention softmax shares the L9 temperature
  softmax structure but with a different meaning (alignment weights vs sampling).

---

## Compile Fix Applied

`\ding{55}` (✗ mark, requires `pifont` — not in the shared preamble) → replaced with
`($\times$)`. `\checkmark` is from amssymb (loaded), kept. Recompiled clean.
**Cyrillic pre-compile scan = 0** (the L9/L10 lesson applied proactively — no Cyrillic
slipped in this time).

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (encoder-decoder/bottleneck; attention; query/key/value; BLEU),
seminal paper, uzbek_angle, locked hand_example (attention softmax), and
`gpu_required: true` all match Day 11 `lecture_official_no: 11`.

---

## Pending

- **Overleaf human review** of prose quality — per `uzbek-course-style` (human gate).
- **P11** (Day 12 — m11 Seq2SeqTranslator) consumes [I3] as its first assert.
- Commit pushed to origin/rtm together with P10's 3 commits (per task instruction).
