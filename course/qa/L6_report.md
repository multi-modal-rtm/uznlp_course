# QA Report: Day 6 Lecture (L6)

**Artifact**: `course/lectures/d06_word2vec.tex` → `d06_word2vec.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 6, `lecture_official_no: 6` — "Neyron tarmoqlarga asoslangan
so'z embeddinglari: Word2Vec (CBOW)"
**Paired practice**: P6 (Day 7 — m06 CustomWord2Vec, gensim CBOW on Uzbek corpus)
**Recap target [B]**: L5 (N-gram/HMM — probability) + P5 (morning autocomplete/POS)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0; `grep "^!"` empty on both passes (MiKTeX x64) |
| Zero `Overfull \hbox` > 10pt | **PASS** | `grep "Overfull \hbox ([1-9][0-9]"` empty |
| Slide count (L1–L5 parity, 4 full cycles) | **PASS** | 48 frames before `\appendix` (footer `/48`); L5 was ~45 — comparable depth. NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] ([H1]–[H4] all present) + [M][N][O][P][Q][R] + [S] appendix |
| [M] Uzbek-language slide (mandatory) | **PASS** | "Agglutinatsiya lug'atni portlatadi — CBOW yordam bera oladimi?" (vocabulary explosion + subword/FastText pointer) |
| Every formula has `\bunda{}` key | **PASS** | [G1] embedding, [G2] CBOW forward, [G3] cross-entropy+grad, [G4] cosine+analogy; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] all `[fragile]` |
| Locked [I2] hand_example | **PASS** | CBOW kirish = avg([0.5,0.3],[0.1,0.7]) = **[0.3, 0.5]** — verbatim from course_map |
| Traceability comment → P6 | **PASS** | [I2] + [Q] carry `# Ma'ruza L6 [I2]-slayd`; [Q] shows `np.allclose(cbow_input, [0.3, 0.5])` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| ASCII apostrophe only (no U+2019) | **PASS** | `grep` count of U+2019 = 0 |
| No BOM / no Cyrillic | **PASS** | File starts `\do…` (no BOM); no Cyrillic codepoints |
| Preamble byte-identical to d05 | **PASS** | Colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim; only title/date/`\textnumero` changed |
| Visual review (rendered PNG @95dpi) | **PASS** | TikZ flow, locked [I2], dense [I3] two-column, 4-col [N] table, [Q] bridge, all code slides — no overflow, no clipping, no collisions |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Numbers (→ P6 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I2]** | `CBOW_input = avg([0.5,0.3],[0.1,0.7])` | **[0.3, 0.5]** | **P6 first assert** (course_map lock) |
| [J2] | `h = avg([0.4,0.2],[0.2,0.6])` | [0.3, 0.4] | task (same shape, diff numbers) |
| [I1] | `cos(one-hot film, kino)` | 0 | motivates dense vectors |
| [J1] | `cos([0.8,0.6],[0.6,0.8])` | 0.96 | dense similarity |
| [I3] | `u(nlp)=0.8 → ŷ(nlp)`, `L`, `∂L/∂u` | 0.690, 0.371, −0.310 | forward/loss/grad (chained from h=[0.3,0.5]) |
| [J3] | `softmax([0,0])`, `L` | 0.5, ln2≈0.693 | task |
| [I4] | `cos([0.8,0.6],[0.6,0.8])` | 0.96 | embedding quality (good) |
| [J4] | `cos([0.8,0.6],[-0.6,0.8])` | 0 | unrelated pair |

All [I]/[J] numbers are mentally computable; only [I2] is locked in course_map.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Frames |
|------------------------|-----------|--------|
| Kirish | [A][B][C][D][E] | 5 (+ TOC) |
| Zich vektor | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | 7 (+ TOC) |
| CBOW | Cycle 2: [F2][G2][H2]+flow-TikZ[I2][J2][K2][L2] | 8 (+ TOC) |
| O'qitish | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | 7 (+ TOC) |
| Baholash | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | 7 (+ TOC) |
| Xulosa | [M][N][O][P][Q][R] | 6 (+ TOC) |
| (appendix [S]) | S1 Skip-gram, S2 negative sampling | 2 (excluded) |

Cycle 2 has one extra slide (the CBOW-flow TikZ diagram) consolidating
forward pass between [H2] derivation and [I2] hand example.

---

## Content Continuity

- **[B] recap** bridges L5 (discrete N-gram tokens, count-based) + this
  morning's P5 (counting can't learn similarity) → motivates learned dense
  vectors.
- **[E] problem-first** shows one-hot orthogonality (cos=0) and L3/P3's poor
  Uzbek pretrained coverage *before* naming Word2Vec.
- **[M] Uzbek**: agglutination → vocabulary explosion
  ('o'rganaman'/'o'rganadi'/'o'rgandim'); CBOW can partially help (shared
  context → near vectors) but treats words as indivisible → pointer to
  subword/FastText.
- **[O] seminal**: Mikolov et al. (2013) *Efficient Estimation…* + discussion
  question (CBOW vs Skip-gram for low-resource Uzbek) due before P6.
- **[Q] bridge**: TikZ pipeline m01 → gensim CBOW → m06 CustomWord2Vec →
  most_similar; first assert = the locked [I2] value.

---

## Deviation from course_map.yaml

None. Topic, 4 subitems, seminal paper, uzbek_angle, locked hand_example
([0.3, 0.5]), and `gpu_required: false` all match Day 6 `lecture_official_no: 6`.

---

## Pending

- **Overleaf human review** of prose quality (sentence-case, natural Uzbek) —
  per `uzbek-course-style` this is a human gate, not automated.
- **P6** (Day 7 — m06 CustomWord2Vec) consumes [I2] as its first assert.
- **w2 milestone** (24-iyun, m01–m05) is the next chronological artifact after
  L6, before P6.
