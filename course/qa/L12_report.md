# QA Report: Day 12 Lecture (L12)

**Artifact**: `course/lectures/d12_transformer.tex` → `d12_transformer.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 12, `lecture_official_no: 12` — "Transformer arxitekturasi va matnni umumlashtirish"
**Paired practice**: P12 (Day 13 — m12 TransformerSummarizer, self-attention + PE, Wikipedia uz maqola-xulosa)
**Recap target [B]**: L11 (Seq2Seq/Attention) + this morning's P11

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0; `Select-String "^!"` empty (clean from first compile — no fixes needed) |
| Zero `Overfull \hbox` > 10pt | **PASS** | `Select-String "Overfull \hbox ([1-9][0-9]"` empty |
| Slide count (L1–L11 parity, 4 full cycles) | **PASS** | **47 frames** before `\appendix` accounting (footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | every cycle has a derivation slide (H1 √d_k, H2 multi-head/PE, H3 residual/norm/mask, H4 P/R/F1) |
| [M] Uzbek-language slide (mandatory) | **PASS** | "Positional encoding va erkin so'z tartibi" (SOV/SVO; self-attention permutation-invariance) |
| Every formula has `\bunda{}` key | **PASS** | [G1] attention, [G2] MHA+PE, [G3] block/FFN, [G4] ROUGE-1; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] all `[fragile]` |
| **Locked [I4] hand_example** | **PASS** | ROUGE-1 `P=1.000, R=0.600, F1=0.750` — verbatim from course_map Day 12 |
| Traceability comment → P12 | **PASS** | [I4] + [Q] carry `# Ma'ruza L12 [I4]-slayd`; [Q] shows `assert abs(f1 - 0.750) < 1e-3` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | pre-compile Cyrillic scan caught 1 slip ("modelга" → "modelga"), fixed; final scan = 0 |
| Preamble byte-identical to d11 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim |
| Visual review (rendered PNG @70dpi) | **PASS** | [I1] self-attn softmax, [G2] MHA+PE box, [I4] locked ROUGE, [K2]/[K3]/[K4] code, [N] table, [Q] bridge — no overflow/clipping/collisions |
| Aux/PNG auto-cleaned | **PASS** | `.aux/.log/.nav/.out/.snm/.toc/.vrb` + `_render12/` removed; only `.tex`+`.pdf` remain |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Numbers (→ P12 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I4]** | ROUGE-1 (ref 5 tok, hyp 3 tok, overlap 3) | **P=1.000, R=0.600, F1=0.750** | **P12 first assert** (course_map lock) |
| [I1] | self-attn `α = softmax([2,1,3])` | [0.245, 0.090, 0.665] | scaled dot-product (ties to L11 [I3]) |
| [J1] | `softmax([1,1,2])` | [0.212, 0.212, 0.576] | task |
| [I2] | `PE(pos=0, d=4)` | [0, 1, 0, 1] | positional encoding |
| [J2] | `PE(pos=1)` first two | [sin 1, cos 1] = [0.841, 0.540] | task |
| [I3] | LayerNorm([1,3]) | [-1, 1] | residual + norm |
| [J3] | LayerNorm([1,2,3]) | [-1.22, 0, 1.22] | task |
| [J4] | ROUGE-1 (overlap 2) | P=0.667, R=0.400, F1=0.500 | task |

Only ROUGE-1 (P/R/F1) is locked by course_map; all other [I]/[J] numbers are mentally / `exp`-table / `sin-cos`-table computable.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L11 (still LSTM-based) + P11; problem = sequential, no parallelism |
| Self-Attention | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | scaled dot-product; **L11 attention generalization** + **[I1] ties to L11 [I3]** |
| Multi-head va PE | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | multi-head + sinusoidal positional encoding |
| Transformer bloki | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | residual + LayerNorm + FFN; enc self-attn, dec masked + cross-attn |
| ROUGE | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | recall-oriented; **locked [I4]** + BLEU-vs-ROUGE contrast |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1 full MHA math][S2 ROUGE-L LCS] |

---

## Content Continuity

- **[B] recap** bridges L11: attention solved the bottleneck, but encoder/decoder were still **LSTM** (sequential) → motivates removing recurrence entirely.
- **[E] problem-first** shows the sequential-compute + weak-long-range cost of RNN/LSTM *before* naming self-attention.
- **Major arc (L11 → L12)**: [F1]/[G1] state explicitly that self-attention is the **generalization** of L11's Bahdanau attention (query now from the sentence itself); **[I1] reuses the exact softmax([2,1,3]) = [0.245,0.090,0.665] from L11 [I3]** to make the continuity concrete.
- **BLEU vs ROUGE**: [F4]/[G4]/[L4] contrast L11's precision-oriented BLEU with L12's recall-oriented ROUGE (summarization).
- **[M] Uzbek**: free word order vs positional encoding — self-attention is permutation-invariant, PE *adds* order without forcing it; argued easier for Uzbek SOV/SVO than RNN.
- **[O] seminal**: Vaswani et al. (2017) "Attention Is All You Need" + discussion (O(n²) cost on long agglutinative Uzbek token streams).
- **[Q] bridge**: TikZ pipeline Wikipedia uz maqola-xulosa → Transformer encoder → decoder → m12 TransformerSummarizer; first assert = the locked [I4] ROUGE-1 F1.

---

## Compile Notes

- **Clean first compile** — zero `^!` errors, zero Overfull >10pt; no package/title-math fixes needed.
- **Cyrillic pre-compile scan** caught one homoglyph slip: `modelга` (Cyrillic "га") → `modelga` ([L2] frame). Fixed before compiling; final scan = 0. (L9/L10 lesson applied.)
- Title-math (`$\sqrt{d_k}$`, `$F_1$`, `$PE_{(pos,2i)}$`) wrapped in `$...$`; comma-bearing titles (`title={So'rov $\cdot$ kalit ...}`, defbox titles) all in `title={...}` braces — L7/L10 pitfalls avoided proactively.
- No `\ding`/`\psmallmatrix` used (L7/L11 pitfalls); `\checkmark` (amssymb) only.

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (self-attention/parallelization; multi-head + positional encoding; abstractive
summarization; ROUGE), seminal paper (Vaswani 2017), uzbek_angle (PE + free word order), and the locked
hand_example (ROUGE-1 P/R/F1) all match Day 12 `lecture_official_no: 12`. `gpu_required: true` pertains
to the paired practice (P12); the lecture is theory — [K] shows self-attention / PE / encoder-block code
that is read, not executed.

---

## Pending

- **Overleaf human review** of prose quality — per `uzbek-course-style` (human gate).
- **P12** (Day 13 — m12 TransformerSummarizer) consumes [I4] (ROUGE-1 F1=0.750) as its first assert.
- Commit pushed to origin/rtm per task instruction.
