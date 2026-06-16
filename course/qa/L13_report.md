# QA Report: Day 13 Lecture (L13)

**Artifact**: `course/lectures/d13_transfer_learning.tex` → `d13_transfer_learning.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 13, `lecture_official_no: 13` — "Transfer Learning va oldindan o'qitilgan modellar (BERT, T5)"
**Paired practice**: P13 (Day 14 — m13 FineTunedClassifier, BERT fine-tuning, Uzum sentiment)
**Recap target [B]**: L12 (Transformer self-attention/MHA/PE) + this morning's P12

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0; `Select-String "^!"` empty |
| Zero `Overfull \hbox` > 10pt | **PASS** | **0 on the first compile** (known K3/K4/title fixes baked in from the start) |
| Slide count (L1–L12 parity, 4 full cycles) | **PASS** | **47 frames** (footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | H1 pre-train/fine-tune, H2 MLM masking, H3 unified text-to-text, H4 HF ecosystem |
| [M] Uzbek-language slide (mandatory) | **PASS** | "WordPiece tokenizatsiya va qo'shimchalar" (mBERT 'o'rganaman'→5 token; XLM-RoBERTa SentencePiece) |
| Every formula has `\bunda{}` key | **PASS** | [G1] sigmoid/BCE, [G2] MLM, [G3] seq CE, [G4] gradient update/warmup; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] all `[fragile]` |
| **Locked [I1] hand_example** | **PASS** | `σ(2.0)≈0.880`, `BCE≈0.128` — verbatim from course_map Day 13 |
| Traceability comment → P13 | **PASS** | [I1] + [Q] carry `# Ma'ruza L13 [I1]-slayd`; [Q] shows `assert abs(loss - 0.128) < 1e-2` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi`; also 0 `musbat\|manfiy` (locked `ijobiy`/`salbiy`) |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | pre-compile Cyrillic scan = 0 (clean from the start) |
| Preamble byte-identical to d12 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim |
| Visual review (rendered PNG @90dpi) | **PASS** | locked [I1], [K4] HF listing (fits), [Q] bridge — no overflow/clipping/collisions |
| Aux/PNG auto-cleaned | **PASS** | `.aux/.log/.nav/.out/.snm/.toc/.vrb` + `_r13/` removed; only `.tex`+`.pdf` remain |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Numbers (→ P13 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I1]** | `σ(2.0)`, BCE (y=1) | **0.880, 0.128** | **P13 first assert** (course_map lock) |
| [J1] | BCE at `σ(0)=0.5` | `-ln 0.5 = 0.693` | task |
| [I2] | MLM `softmax([1,3,2])` | [0.090, 0.665, 0.245] → "olma" | masked-token prediction |
| [J2] | `softmax([1,1,2])` | [0.212, 0.212, 0.576] | task |
| [I3] | T5 token CE, `p=0.665` | `-ln 0.665 = 0.408` | seq cross-entropy |
| [J3] | token CE, `p=0.245` | `-ln 0.245 = 1.406` | task |
| [I4] | fine-tune steps (5000, b=16, 3 ep) | `⌈5000/16⌉×3 = 939` | training-step count |
| [J4] | steps (8000, b=32, 2 ep) | `250×2 = 500` | task |

Only the BCE/sigmoid ([I1]) is locked by course_map; all other [I]/[J] numbers are `exp`/`ln`-table or arithmetic computable.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L12 (still needs huge data) + P12; problem = scratch training impossible for low-resource Uzbek |
| Transfer | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | pre-training + fine-tuning; sigmoid/BCE; **locked [I1]** |
| BERT | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | bidirectional encoder, MLM, [CLS] |
| T5 | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | text-to-text framing, span-corruption |
| HF | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | transformers/datasets/Trainer; fine-tune workflow |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1 80/10/10 masking][S2 T5 span-corruption] |

---

## Content Continuity

- **[B] recap** bridges L12: the Transformer is powerful but training it from scratch needs huge data/compute → unavailable for Uzbek → motivates transfer learning.
- **[E] problem-first** shows scratch-training infeasibility for low-resource Uzbek *before* naming pre-train/fine-tune.
- **Arc (L12 → L13)**: explicitly states BERT = Transformer **encoder** and T5 = Transformer **encoder-decoder** (L12 architecture, pre-trained). Transfer learning = "not from scratch — pre-trained" ([B]/[F1]/[E]).
- **[M] Uzbek**: mBERT WordPiece fragments Uzbek suffixes ('o'rganaman' → ~5 subwords); XLM-RoBERTa SentencePiece keeps Uzbek words more intact — affects fine-tuning quality.
- **[O] seminal**: Devlin et al. (2019) BERT + discussion (mBERT subword fragmentation impact on Uzbek sentiment; XLM-R).
- **[Q] bridge**: TikZ pipeline Uzum sentiment → DistilBERT `from_pretrained` → Trainer fine-tune → m13 FineTunedClassifier; first assert = the locked [I1] BCE.

---

## Compile Notes

This deck was authored with the compile fixes from the prior attempt baked in from the start, so it
compiled **clean on the first pass** (0 errors, 0 Overfull >10pt):
- **Title** given a `\\` break (`...modellar\\ (BERT, T5)`) → clean two-line title slide (avoids the
  ~14pt title overflow a single-line long title produced).
- **[K3] T5 listing** uses two plain `from transformers import ...` lines (no deep paren-aligned
  continuation, which had created an unbreakable 26-space overflow).
- **[K4] HF listing** code column widened to 0.66 with 2-space continuation indent; mapping table set to
  `\scriptsize` with `p{2.0cm}p{1.5cm}` and shortened entries (`num_labels`, `2 sinf`).
- **Section short-titles** kept short (`[Transfer]`, `[HF]`) so footer navigation never overflows.
- **Cyrillic pre-compile scan = 0** — no homoglyph slipped in (the recurring L9/L10/L12 defect avoided).

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (transfer learning pre-train/fine-tune; BERT; T5; Hugging Face), seminal paper
(Devlin 2019), uzbek_angle (WordPiece + suffixes, mBERT vs XLM-R), and the locked hand_example
(sigmoid/BCE) all match Day 13 `lecture_official_no: 13`. `gpu_required: true` pertains to the paired
practice (P13); the lecture is theory — [K] shows HF `transformers` code that is read, not executed.

---

## Pending

- **Overleaf human review** of prose quality — per `uzbek-course-style` (human gate).
- **P13** (Day 14 — m13 FineTunedClassifier) consumes [I1] (BCE=0.128) as its first assert.
- ⚠️ Unpushed: P12 (4 commits) + this L13 commit — origin/rtm currently at `f5ee38b`.
