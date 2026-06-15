# QA Report — L5: Ehtimollik modellari — N-grammalar va so'z turkumlari

**Date:** 2026-06-15
**Artifact:** `course/lectures/d05_til_modellari.tex` (+ compiled `.pdf`)
**Lecture:** Official topic №5 — "Ehtimollik modellari: N-grammalar va so'z turkumlarini teglash"
**Delivered:** Day 5 afternoon (22-iyun 2026, 11:00–12:20, A.A. Abdulali)
**Paired practice:** P5 (Day 6 — m05 Autocomplete + m05b POSTagger)
**Continuity:** L4 (qidiruv/imlo) → L5; morning P4 referenced in [B]
**Position:** first lecture of Week 2 (klassik pipeline → ehtimollik modellari)

---

## Gate Results

| Gate | Result | Notes |
|---|---|---|
| Terminology grep (`professor\|talaba\|student\|o'qituvchi`) | **PASS** | 0 defects |
| Slide count (rendered, body only) | **PASS (4-subitem)** | 45 — L1–L4 parity (4 full cycles), as instructed |
| Self-contained (.tex) | **PASS** | No `\input`/`\include`/`\includegraphics`/`\bibliography`/`\setmainfont` |
| Archetype structure [A]–[S] | **PASS** | 4 FULL cycles ([F]–[L] each); [M] mandatory Uzbek slide |
| `\bunda{}` on every formula [G] | **PASS** | G1, G2 (×2 defbox), G3, G4 + S2 |
| `[fragile]` on every lstlisting frame | **PASS** | 4 lstlisting frames (K1, K2, K3, K4) all `[fragile]` |
| Environment balance | **PASS** | frame 41/41; columns/tcolorbox/tikzpicture/lstlisting matched |
| Encoding hygiene (ASCII apostrophe; no U+2019/Cyrillic/BOM) | **PASS** | all `'` ASCII; T1-safe |
| **pdflatex compile ×2** | **PASS** | MiKTeX (local) — **0 `^!` errors**, **0 `Overfull \hbox (>10pt)`** |
| Visual PDF review (60 pages) | **PASS** | key/new frames inspected at 95 dpi; 3 defects found & fixed (below) |
| Locked [I4] traceability value | **PASS** | `δ(VB,t=2)=0.3402`, sequence `[NN, VB]` — matches course_map hand_example |

---

## Post-Compile Defects Found & Fixed

| # | Frame | Defect | Fix |
|---|---|---|---|
| 1 | [L4] | tcolorbox title typo `title=Underflow}]` (stray `}`) | removed stray brace |
| 2 | [I2], [I4] | tcolorbox titles with **internal commas** parsed as pgfkeys keys (fatal) | braced titles `title={...}` |
| 3 | [E] title, [M] prose | `men ___` — three underscores in **text mode** → math-subscript error (8 cascading `Missing $/{/}` each) | replaced with `\underline{\hspace{0.6cm}}` blank |

After fixes: clean compile (0 `^!`, 0 Overfull >10pt). The locked [I4] Viterbi
trellis, all 4 code frames (K1–K4), derivations and tables re-verified visually —
no overflow, no artifacts.

---

## Slide-Count Breakdown

| Metric | Value |
|---|---|
| `\begin{frame}` in source (incl. 1 macro, 2 appendix) | 41 |
| Content frames before `\appendix` | 38 |
| `\maketitle` | +1 |
| `\section` count | 6 |
| AtBeginSection firings | +6 |
| **Rendered logical slides (body)** | **45** |
| Total frames (`\inserttotalframenumber`) | 47 (45 + 2 appendix) |
| `\pause` overlays | ~13 (B, H1–H4 ×2, J1–J4) |
| Physical PDF pages | 60 |
| Appendix [S] frames (excluded) | 2 (S1, S2) |

---

## Sub-Item Coverage (4 official sub-items) — full cycles

| Official sub-item | Cycle (archetypes) |
|---|---|
| 1. N-grammalar (uni/bi/trigram) + avtomatik to'ldirish | §2: F1·G1·H1·I1·J1·K1·L1 (7) |
| 2. Perplexity + smoothing (Laplace/add-k) | §3: F2·G2·H2·I2·J2·K2·L2 (7) |
| 3. So'z turkumlari (POS) + Markov zanjirlari | §4: F3·G3·H3·I3·J3·K3·L3 (7) |
| 4. Yashirin Markov modeli + Viterbi | §5: F4·G4·H4·I4★·J4·K4·L4 (7) |

★ [I4] carries the locked P5-traceability number.

---

## Archetype Completeness

| Archetype | Present | Notes |
|---|---|---|
| [A] Title | ✓ | `\maketitle` |
| [B] Recap quiz (2 Q + `\pause` + 2 A) | ✓ | L4 noisy-channel P(w)=unigram + P4 DP → bugun DP (Viterbi) |
| [C] Measurable objectives | ✓ | 4 verbs: keltirib chiqara / hisoblay / qo'llay / keltirib chiqara |
| [D] Plan + time-budget | ✓ | 4 cycles, 80 min; "Week 2 ochiladi" |
| [E] Problem-first | ✓ | "men ___" autocomplete + POS teglash |
| [F] Intuition | ✓ | F1 N-gram, F2 perplexity/smoothing, F3 POS/Markov, F4 HMM |
| [G] Defbox + `\bunda` | ✓ | G1 bigram MLE, G2 perplexity+Laplace, G3 Markov, G4 HMM/Viterbi |
| [H] Derivation | ✓ | H1 chain→Markov, H2 Laplace +V, H3 Markov, H4 Viterbi DP |
| [I] Hand example | ✓ | I1 P(kitob\|men)=2/3, I2 add-1=1/7, I3 P(NN,VB)=0.42, I4★ δ=0.3402 |
| [J] Your-turn warnbox→`\pause`→okbox | ✓ | J1 1/3, J2 1/3 (add-1), J3 0.09, J4 δ(NN)=0.0252 |
| [K] Code ↔ formula bridge | ✓ | K1 bigram, K2 add-1+perplexity, K3 Markov, K4 Viterbi |
| [L] Pitfall + real error | ✓ | L1 zero problem, L2 perplexity comparability, L3 long-dependency, L4 underflow→log |
| [M] Uzbek-language (mandatory) | ✓ | SOV (kesim oxirida) → uzoq bog'liqlik; agglutinatsiya → N-gram lug'at portlashi |
| [N] Synthesis comparison table | ✓ | unigram/bigram/trigram/HMM — kontekst vs ma'lumot murosasi |
| [O] Seminal paper + discussion Q | ✓ | Rabiner (1989); Q on Uzbek tagged-corpus scarcity |
| [P] Objectives checkmarked | ✓ | `\bajarildi` ×4 |
| [Q] Bridge to practice (TikZ) | ✓ | → m05 Autocomplete + m05b POSTagger; P5 first assert |
| [R] References | ✓ | 5 (Rabiner, J&M, Shannon, Chen-Goodman, nltk.lm) |
| [S] Appendix backups | ✓ | S1 log-space Viterbi, S2 perplexity↔entropy |

---

## Locked Hand-Example ([I4]) — P5 Traceability

Verbatim from course_map.yaml `hand_example` (Day 5):
- States NN, VB; observations nlp (t=1), yozdi (t=2).
- π(NN)=0.7, π(VB)=0.3; B(nlp|NN)=0.9, B(nlp|VB)=0.1, B(yozdi|NN)=0.1, B(yozdi|VB)=0.9;
  A(VB|NN)=0.6, A(NN|VB)=0.3, A(NN|NN)=0.4, A(VB|VB)=0.7.
- t=1: δ(NN)=0.63, δ(VB)=0.03. t=2: δ(VB)=max(0.378, 0.021)·0.9 = **0.3402**.
- Backtrace: **[NN, VB]**.

P5 first assert (Day 6): `assert abs(delta_VB_t2 - 0.3402) < 1e-4` and `path == ["NN","VB"]`.
Traceability comment in .tex ([I4]/[Q]): `# Ma'ruza L5 [I4]-slayd`.

Other cycle hand-examples (designed mentally-computable, consistent with topic):
- [I1] P(kitob|men)=2/3≈0.667; [J1] P(non|men)=1/3.
- [I2] add-1 P(o'qidim|non)=1/7≈0.143 (unsmoothed=0); [J2] add-1 P(kitob|men)=3/9=1/3.
- [I3] P(NN,VB)=π(NN)·A(VB|NN)=0.42; [J3] P(VB,NN)=0.09.
- [J4] δ(NN,t=2)=0.0252 (VB wins).

---

## Content Notes

- **GPU**: `gpu_required: false` (L5). Lecture only; CPU theme.
- **Seminal paper**: Rabiner (1989) — matches `seminal_paper` in course_map.yaml.
- **Uzbek angle** [M]: SOV word order (verb last) → long-dependency that bigram misses;
  agglutination → N-gram vocabulary explosion / sparsity → smoothing critical. Matches `uzbek_angle`.
- **Hand-example design**: only the Viterbi [I4] is locked in course_map; cycles 1–3 hand
  examples were designed here (clean, mentally checkable) on a 3-sentence Uzbek corpus
  (`men kitob o'qidim` / `men non yedim` / `men kitob oldim`, |V|=6). No forbidden terms.
- **Style**: L1 etalon — sentence case, first-person plural, full-statement titles, ASCII
  apostrophe. Preamble copied verbatim from d04 (identical visual identity).

---

**Status: PASS (all gates incl. compile)** — terminology, archetypes (4 full cycles),
`\bunda{}`, fragile frames, environment balance, encoding hygiene, and locked [I4]
traceability all clear. Compile **PASS locally** (MiKTeX, pdflatex ×2): 0 `^!`, 0 Overfull >10pt.
45 rendered slides (L1–L4 parity). 3 defects found and fixed; key frames visually
inspected — no overflow. Compiled `course/lectures/d05_til_modellari.pdf` committed
alongside the source.
