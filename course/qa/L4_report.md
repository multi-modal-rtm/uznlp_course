# QA Report — L4: Masofaga asoslangan qidiruv va imlo tuzatish

**Date:** 2026-06-15
**Artifact:** `course/lectures/d04_qidiruv_imlo.tex` (+ compiled `.pdf`)
**Lecture:** Official topic №4 — "Masofaga asoslangan qidiruv va imlo tuzatish usullari"
**Delivered:** Day 4 afternoon (19-iyun 2026, 11:00–12:20, A.A. Abdulali)
**Paired practice:** P4 (Day 5, 22-iyun 2026, I.R. Atadjanov — `SpellLSHRetriever`, m04)
**Continuity:** L3 (embeddinglar) → L4; morning P3 (PretrainedEmbedder) referenced in [B]
**Position:** final lecture of Week 1 — [N] synthesizes the m01→m02→m03→m04 arc

---

## Gate Results

| Gate | Result | Notes |
|---|---|---|
| Terminology grep (`professor\|talaba\|student\|o'qituvchi`) | **PASS** | 0 defects (apostrophe examples use `yo'l`/`to'g'ri`, not `o'qituvchi`) |
| Slide count (rendered, body only) | **PASS** | 27 logical slides — within standard 22–28 (no exception needed) |
| Self-contained (.tex) | **PASS** | No `\input`/`\include`/`\includegraphics`/`\bibliography`/`\setmainfont` |
| Archetype structure [A]–[S] | **PASS** | All present; [M] mandatory Uzbek slide included |
| `\bunda{}` on every formula [G] | **PASS** | G1 (Jaccard), G2 (noisy channel), G3 (DP) + S2 appendix |
| `[fragile]` on every lstlisting frame | **PASS** | 1 lstlisting frame (K3) is `[fragile]` |
| Environment balance | **PASS** | frame 25/25, columns/tcolorbox/tikzpicture/lstlisting all matched |
| Encoding hygiene (ASCII apostrophe; no U+2019/Cyrillic/BOM) | **PASS** | all `'` are ASCII; T1-safe |
| **pdflatex compile ×2** | **PASS** | MiKTeX (local) — 0 `^!` errors, 0 `Overfull \hbox (>10pt)` |
| Visual PDF review (33 pages) | **PASS** | all pages rendered at 95 dpi and inspected; 1 defect found & fixed (below) |
| Locked [I3] traceability value | **PASS** | `edit_distance("qo'l","ko'l")=1`, `("dastur","dastir")=1` — matches course_map hand_example |

---

## Post-Compile Visual Review — Defect Found & Fixed

| # | Frame | Defect | Fix |
|---|---|---|---|
| 1 | [I3] | A stray leftover header `tabular` rendered as a misplaced `"" k o ' l` fragment below-left of the DP table | Removed the stray first `tabular`; the second (complete) DP table already has the correct header row |

After the fix the deck recompiles clean (0 `^!`, 0 Overfull >10pt) and the I3 DP
table renders correctly (`D[4][4]=1` highlighted). All other pages were clean on
first compile (G4 5-box architecture pipeline, Q bridge pipeline, S1 7×7 DP table,
G2 noisy-channel argmax with underbraces — all fit width, no overflow).

---

## Slide-Count Breakdown

| Metric | Value |
|---|---|
| `\begin{frame}` in source (incl. 1 in `\AtBeginSection` macro, 2 appendix) | 25 |
| Authored frames (excl. macro) | 24 |
| Content frames before `\appendix` | 22 |
| `\maketitle` | +1 |
| `\section` count | 4 |
| AtBeginSection "Prezentatsiya rejasi" firings | +4 |
| **Rendered logical slides (body)** | **27** |
| `\pause` overlays (body) | 4 (B, H3×2, J3) |
| Physical PDF pages (body+appendix+pauses) | 33 |
| Appendix [S] frames (excluded) | 2 (S1, S2) |

L4 fits the **standard 22–28 ceiling** (27) — unlike L1/L2/L3 which used the
4-subitem exception. Achieved by compressing per-cycle archetypes: [F] once (F1),
intuitions folded into [G] lead-ins, and subitem 4 covered by a single
architecture slide (G4) rather than a full cycle.

---

## Sub-Item Coverage (4 official sub-items)

| Official sub-item | Slides |
|---|---|
| 1. Samarali o'xshashlik qidiruvi: KNN, LSH | [F1][G1][I1] (§2) |
| 2. Noisy channel imlo tuzatish modeli | [G2][I2] (§3) |
| 3. Minimal tahrir (Levenshtein) masofasi va DP | [G3][H3][I3★][J3][K3][L3] (§3) |
| 4. Avtomatik tuzatish tizimi arxitekturasi | [G4] (§3) |

★ [I3] carries the locked P4-traceability numbers.

---

## Per-Slide Archetype Coverage Table

| # | Archetype | Frame title (sentence case) | Sub-item |
|---|---|---|---|
| 1 | **[A]** | *(title — `\maketitle`)* | — |
| 2 | *(AtBegSec)* | Prezentatsiya rejasi (Kirish) | — |
| 3 | **[B]** | Takrorlash: o'xshashlikni qanday o'lchagandik? | — |
| 4 | **[C]** | Siz bugungi dars oxirida quyidagilarni bajara olasiz | — |
| 5 | **[D]** | Bugungi dars rejasi | — |
| 6 | **[E]** | Muammo: "telfon" deb yozsa, aniq moslik topilmaydi | — |
| 7 | *(AtBegSec)* | Prezentatsiya rejasi (Tez qidiruv) | — |
| 8 | **[F1]** | Intuitsiya: o'xshashlarni "savatlarga" ajratamiz | 1 |
| 9 | **[G1]** | Jaccard o'xshashligi, MinHash va LSH: rasmiy ta'rif | 1 |
| 10 | **[I1]** | Qo'lda hisob: ikki hujjatning Jaccard o'xshashligi | 1 |
| 11 | *(AtBegSec)* | Prezentatsiya rejasi (Imlo tuzatish) | — |
| 12 | **[G2]** | Noisy channel modeli: eng ehtimoliy to'g'ri so'z | 2 |
| 13 | **[I2]** | Qo'lda hisob: "kitop" ni qanday tuzatamiz? | 2 |
| 14 | **[G3]** | Tahrir masofasi (Levenshtein): rasmiy ta'rif | 3 |
| 15 | **[H3]** | Keltirib chiqarish: nega aynan shu uch had? | 3 |
| 16 | **[I3] ★** | Qo'lda hisob: edit_distance("qo'l","ko'l") = 1 | 3 |
| 17 | **[J3]** | Sizning vazifangiz: tahrir masofasini toping | 3 |
| 18 | **[K3]** | Kod ↔ formula: tahrir masofasi DP da | 3 |
| 19 | **[L3]** | Cheklovlar va tipik xatolar | 3 |
| 20 | **[G4]** | Avtomatik tuzatish tizimi: komponentlar va oqim | 4 |
| 21 | *(AtBegSec)* | Prezentatsiya rejasi (Xulosa) | — |
| 22 | **[M]** | O'zbek tilida: apostrof va qo'shimchalar masofani chalg'itadi | All |
| 23 | **[N]** | Sintez: 1-haftada o'xshashlikni uch darajada o'lchadik | 1–3 |
| 24 | **[O]** | Seminal manba: Norvig (2009) | All |
| 25 | **[P]** | Bugungi maqsadlar bajarildi | All |
| 26 | **[Q]** | Ertaga P4: imlo tuzatish va LSH qidiruvini qurasiz | All |
| 27 | **[R]** | Adabiyotlar | — |
| *(28)* | **[S1]** *(appendix)* | "dastur"/"dastir" to'liq DP jadvali | 3 |
| *(29)* | **[S2]** *(appendix)* | MinHash va LSH banding matematikasi | 1 |

---

## Archetype Completeness

| Archetype | Present | Notes |
|---|---|---|
| [A] Title | ✓ | `\maketitle` |
| [B] Recap quiz (2 Q + `\pause` + 2 A) | ✓ | L3 cosine + this-morning P3 `most_similar` O(N) → motivates LSH |
| [C] Measurable objectives | ✓ | 4 verbs: keltirib chiqara / hisoblay / qo'llay / taqqoslay |
| [D] Plan + time-budget | ✓ | 4 blocks, 80 min; notes "Week 1 finale" |
| [E] Problem-first | ✓ | "telfon"→no exact match + O(N) slow search |
| [F] Intuition | ✓ | F1 (LSH buckets) |
| [G] Defbox + `\bunda` | ✓ | G1 Jaccard/MinHash/LSH, G2 noisy channel, G3 Levenshtein DP, G4 architecture |
| [H] Derivation | ✓ | H3 (DP recurrence: 3 cases + base) |
| [I] Hand example | ✓ | I1 Jaccard=0.5, I2 kitop→kitob, I3★ edit=1 |
| [J] Your-turn warnbox→`\pause`→okbox | ✓ | J3 (edit distances: substitution + insertion) |
| [K] Code ↔ formula bridge | ✓ | K3 edit_distance DP code → maps to P4 m04 |
| [L] Pitfall + real error | ✓ | L3 (missing DP base row; LSH false negatives) |
| [M] Uzbek-language (mandatory) | ✓ | apostrophe variants inflate edit distance; agglutination → OOV |
| [N] Synthesis comparison table | ✓ | Levenshtein/Jaccard+LSH/cosine + Week-1 module arc |
| [O] Seminal paper + discussion Q | ✓ | Norvig (2009); Q on Uzbek agglutinative candidates |
| [P] Objectives checkmarked | ✓ | `\bajarildi` ×4 |
| [Q] Bridge to practice (TikZ) | ✓ | → m04 SpellLSHRetriever; P4 first assert snippet |
| [R] References | ✓ | 5 (Norvig, J&M, Levenshtein, Broder/Indyk-Motwani, datasketch) |
| [S] Appendix backups | ✓ | S1 full DP table, S2 LSH banding math |

---

## Locked Hand-Example ([I3]) — P4 Traceability

Verbatim from course_map.yaml `hand_example`:
- `edit_distance("qo'l", "ko'l") = 1` (q→k substitution; full 5×5 DP table shown)
- `edit_distance("dastur", "dastir") = 1` (u→i substitution; 7×7 table in appendix S1)

P4 first assert (Day 5): `assert sp.edit_distance("qo'l","ko'l") == 1`
Traceability comment in .tex ([I3] / [Q]): `# Ma'ruza L4 [I3]-slayd`

Additional numeric values for P4 alignment:
- [I1] Jaccard(A,B) = 2/4 = 0.5 (contrast L3 cosine 2/3 on same docs)
- [J1]/[I2] noisy channel: `kitop`→`kitob`, score 0.080 > 0.002
- [J3] edit("maktab","maktub")=1; edit("kitob","kitoblar")=3

---

## Content Notes

- **GPU**: `gpu_required: false` (L4). Lecture only; CPU-only theme.
- **Seminal paper**: Norvig (2009) — matches `seminal_paper` in course_map.yaml.
- **Uzbek angle** [M]: apostrophe (`o'`,`g'`) variants + ASCII vs U+2019 inflate
  edit distance; agglutination enlarges candidate space / OOV. Matches `uzbek_angle`.
  Examples chosen (`yo'l`/`yol`, `to'g'ri`/`togri`) deliberately avoid the
  forbidden-as-audience term `o'qituvchi` while making the same point.
- **Style**: follows manually-polished L1 etalon — Uzbek sentence case, first-person
  plural, full-statement frame titles, ASCII apostrophe throughout. (L2 still Title
  Case from before the rule; L3 and L4 follow the L1 etalon.)
- **AtBeginSection title** uses "Prezentatsiya rejasi" (L1/L3 etalon).

---

**Status: PASS (all gates incl. compile)** — terminology, archetypes, `\bunda{}`,
fragile frames, environment balance, encoding hygiene, slide count (27, standard
ceiling), and locked [I3] traceability all clear. Compile **PASS locally** (MiKTeX,
pdflatex ×2): 0 `^!`, 0 Overfull >10pt. All 33 rendered pages visually inspected;
1 defect found and fixed. Compiled `course/lectures/d04_qidiruv_imlo.pdf` committed
alongside the source.
