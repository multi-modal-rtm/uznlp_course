# QA Report — L4: Masofaga asoslangan qidiruv va imlo tuzatish

**Date:** 2026-06-15
**Artifact:** `course/lectures/d04_qidiruv_imlo.tex` (+ compiled `.pdf`)
**Lecture:** Official topic №4 — "Masofaga asoslangan qidiruv va imlo tuzatish usullari"
**Delivered:** Day 4 afternoon (19-iyun 2026, 11:00–12:20, A.A. Abdulali)
**Paired practice:** P4 (Day 5, 22-iyun 2026, I.R. Atadjanov — `SpellLSHRetriever`, m04)
**Continuity:** L3 (embeddinglar) → L4; morning P3 (PretrainedEmbedder) referenced in [B]
**Position:** final lecture of Week 1 — [N] synthesizes the m01→m02→m03→m04 arc

> **Revision note:** initial L4 was produced at the standard 22–28 ceiling (27
> slides). On human request it was **expanded to full L1–L3 depth (42 slides)** —
> every cycle now has its own [F]/[J]/[K]/[L], cycle 4 gains [I4]/[J4]. Levenshtein
> core and the locked [I3] are unchanged.

---

## Gate Results

| Gate | Result | Notes |
|---|---|---|
| Terminology grep (`professor\|talaba\|student\|o'qituvchi`) | **PASS** | 0 defects (apostrophe examples use `yo'l`/`to'g'ri`, not `o'qituvchi`) |
| Slide count (rendered, body only) | **PASS (4-subitem exception)** | 42 — matches L1 (48)/L2 (43)/L3 (40); follows the documented 4-subitem-lecture exception, per human request for parity |
| Self-contained (.tex) | **PASS** | No `\input`/`\include`/`\includegraphics`/`\bibliography`/`\setmainfont` |
| Archetype structure [A]–[S] | **PASS** | All present; 4 full cycles; [M] mandatory Uzbek slide |
| `\bunda{}` on every formula [G] | **PASS** | G1 (Jaccard), G2 (noisy channel), G3 (DP) + S2 appendix; G4 is architecture (no formula → no `\bunda`) |
| `[fragile]` on every lstlisting frame | **PASS** | 3 lstlisting frames (K1, K2, K3) all `[fragile]` — clean compile confirms |
| Environment balance | **PASS** | frame begin/end matched; columns/tcolorbox/tikzpicture/lstlisting matched |
| Encoding hygiene (ASCII apostrophe; no U+2019/Cyrillic/BOM) | **PASS** | all `'` are ASCII; T1-safe |
| **pdflatex compile ×2** | **PASS** | MiKTeX (local) — 0 `^!` errors, 0 `Overfull \hbox (>10pt)` |
| Visual PDF review (55 pages) | **PASS** | key/new frames inspected at 95 dpi (K1/K2 code, I4 trace table, F2 channel TikZ, I3 DP table); no overflow, no artifacts |
| Locked [I3] traceability value | **PASS** | `edit_distance("qo'l","ko'l")=1`, `("dastur","dastir")=1` — matches course_map hand_example |

---

## Slide-Count Breakdown

| Metric | Value |
|---|---|
| `\begin{frame}` in source (incl. 1 macro, 2 appendix) | 38 |
| Authored frames (excl. macro) | 37 |
| Content frames before `\appendix` | 35 |
| `\maketitle` | +1 |
| `\section` count | 6 |
| AtBeginSection "Prezentatsiya rejasi" firings | +6 |
| **Rendered logical slides (body)** | **42** |
| Total frames (`\inserttotalframenumber`) | 44 (42 body + 2 appendix) |
| `\pause` overlays | 11 (B, H1×2, H2×2, H3×2, J1, J2, J3, J4) |
| Physical PDF pages | 55 |
| Appendix [S] frames (excluded from count) | 2 (S1, S2) |

---

## Sub-Item Coverage (4 official sub-items) — full cycles

| Official sub-item | Cycle (archetypes) |
|---|---|
| 1. Samarali o'xshashlik qidiruvi: KNN, LSH | §2: F1·G1·H1·I1·J1·K1·L1 (7) |
| 2. Noisy channel imlo tuzatish modeli | §3: F2·G2·H2·I2·J2·K2·L2 (7) |
| 3. Minimal tahrir (Levenshtein) masofasi va DP | §4: F3·G3·H3·I3★·J3·K3·L3 (7) |
| 4. Avtomatik tuzatish tizimi arxitekturasi | §5: F4·G4·I4·J4 (4) |

★ [I3] carries the locked P4-traceability numbers. Cycle 4 follows L1's lighter
cycle-4 pattern (F·G·I·J; no H/K/L) — architecture, not a computational method.

---

## Archetype Completeness

| Archetype | Present | Notes |
|---|---|---|
| [A] Title | ✓ | `\maketitle` |
| [B] Recap quiz (2 Q + `\pause` + 2 A) | ✓ | L3 cosine + this-morning P3 `most_similar` O(N) → motivates LSH |
| [C] Measurable objectives | ✓ | 4 verbs: keltirib chiqara / hisoblay / qo'llay / taqqoslay |
| [D] Plan + time-budget | ✓ | 4 blocks, 80 min; "Week 1 finale" |
| [E] Problem-first | ✓ | "telfon"→no exact match + O(N) slow search |
| [F] Intuition | ✓ | F1 (LSH buckets), F2 (noisy channel), F3 (edit distance), F4 (system) |
| [G] Defbox + `\bunda` | ✓ | G1 Jaccard/MinHash/LSH, G2 noisy channel, G3 Levenshtein DP, G4 architecture |
| [H] Derivation | ✓ | H1 (MinHash collision = Jaccard), H2 (Bayes), H3 (DP recurrence) |
| [I] Hand example | ✓ | I1 Jaccard=0.5, I2 kitop→kitob, I3★ edit=1, I4 telfon→telefon trace |
| [J] Your-turn warnbox→`\pause`→okbox | ✓ | J1 Jaccard=0.2, J2 maktap→maktab, J3 edit distances, J4 pipeline trace |
| [K] Code ↔ formula bridge | ✓ | K1 datasketch MinHash/LSH, K2 correct(), K3 edit_distance DP |
| [L] Pitfall + real error | ✓ | L1 (LSH false neg/S-curve), L2 (candidate gen / ignore P(w)), L3 (DP base row) |
| [M] Uzbek-language (mandatory) | ✓ | apostrophe variants inflate edit distance; agglutination → OOV |
| [N] Synthesis comparison table | ✓ | Levenshtein/Jaccard+LSH/cosine + Week-1 module arc |
| [O] Seminal paper + discussion Q | ✓ | Norvig (2009); Q on Uzbek agglutinative candidates |
| [P] Objectives checkmarked | ✓ | `\bajarildi` ×4 |
| [Q] Bridge to practice (TikZ) | ✓ | → m04 SpellLSHRetriever; P4 first assert snippet |
| [R] References | ✓ | 5 (Norvig, J&M, Levenshtein, Broder/Indyk-Motwani, datasketch) |
| [S] Appendix backups | ✓ | S1 full 7×7 DP table, S2 LSH banding math |

---

## Locked Hand-Example ([I3]) — P4 Traceability

Verbatim from course_map.yaml `hand_example`:
- `edit_distance("qo'l", "ko'l") = 1` (q→k substitution; full 5×5 DP table shown)
- `edit_distance("dastur", "dastir") = 1` (u→i substitution; 7×7 table in appendix S1)

P4 first assert (Day 5): `assert sp.edit_distance("qo'l","ko'l") == 1`
Traceability comment in .tex ([I3] / [Q]): `# Ma'ruza L4 [I3]-slayd`

Additional numeric values for P4 alignment:
- [I1] Jaccard(A,B) = 2/4 = 0.5; [J1] Jaccard(A,C) = 1/5 = 0.2 (contrast L3 cosine 2/3)
- [H1] `P[minhash(A)=minhash(B)] = |A∩B|/|A∪B| = J(A,B)`
- [I2] noisy channel: `kitop`→`kitob`, 0.080 > 0.002; [J2] `maktap`→`maktab`, 0.14 > 0.01
- [I4] `telfon`→`telefon` (0.27 > 0.010); [J3] edit("maktab","maktub")=1, edit("kitob","kitoblar")=3
- [S2] LSH candidate prob `P(s) = 1 − (1 − s^r)^b`

---

## Content Notes

- **GPU**: `gpu_required: false` (L4). Lecture only; CPU-only theme.
- **Seminal paper**: Norvig (2009) — matches `seminal_paper` in course_map.yaml.
- **Uzbek angle** [M]: apostrophe (`o'`,`g'`) variants + ASCII vs U+2019 inflate
  edit distance; agglutination enlarges candidate space / OOV. Matches `uzbek_angle`.
  Examples (`yo'l`/`yol`, `to'g'ri`/`togri`) deliberately avoid the
  forbidden-as-audience term `o'qituvchi` while making the same point.
- **Style**: follows manually-polished L1 etalon — Uzbek sentence case, first-person
  plural, full-statement titles, ASCII apostrophe throughout. (L2 still Title Case;
  L3 and L4 follow the L1 etalon.)
- **AtBeginSection title** uses "Prezentatsiya rejasi" (L1/L3 etalon). 6 sections.

---

**Status: PASS (all gates incl. compile)** — terminology, archetypes (4 full cycles),
`\bunda{}`, fragile frames, environment balance, encoding hygiene, and locked [I3]
traceability all clear. Compile **PASS locally** (MiKTeX, pdflatex ×2): 0 `^!`, 0
Overfull >10pt. 42 rendered slides (L1–L3 parity, 4-subitem exception per human
request). Key/new frames visually inspected — no overflow. Compiled
`course/lectures/d04_qidiruv_imlo.pdf` committed alongside the source.
