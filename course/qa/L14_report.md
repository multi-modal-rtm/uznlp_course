# QA Report: Day 14 Lecture (L14)

**Artifact**: `course/lectures/d14_rag.tex` → `d14_rag.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 14, `lecture_official_no: 14` — "RAG va vektor ma'lumotlar bazalari"
**Paired practice**: P14 (Day 15 — m14 RAGEngine, chunk+embed+FAISS+generate, uz_kb)
**Recap target [B]**: L13 (transfer learning, BERT/T5, fine-tuning) + this morning's P13

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0; `Select-String "^!"` empty |
| Zero `Overfull \hbox` > 10pt | **PASS** | `Select-String "Overfull \hbox ([1-9][0-9]"` empty (0) |
| Slide count (L1–L13 parity, 4 full cycles) | **PASS** | **47 frames** (footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | H1 RAG-vs-fine-tune, H2 prompt budget, H3 cosine, H4 ANN |
| [M] Uzbek-language slide (mandatory) | **PASS** | "RAG va huquqiy matnlar (lex.uz)" (hallucination-prone legal domain; multilingual embedding) |
| Every formula has `\bunda{}` key | **PASS** | [G1] RAG, [G2] token budget, [G3] cosine/top-k, [G4] index/upsert/query; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] all `[fragile]` |
| **Locked [I2] hand_example** | **PASS** | prompt `3×200+100 = 700` token; `700/128000 ≈ 0.0055 < 1%`; retrieved_docs=3 — verbatim from course_map |
| Traceability comment → P14 | **PASS** | [I2] + [Q] carry `# Ma'ruza L14 [I2]-slayd`; [Q] shows `assert prompt_tokens == 700` / `assert retrieved_docs == 3` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | pre-compile scan caught **8** homoglyphs (modelга, manbани, uzunlikка/дan, ozгina, tezlikка, manbaга), all fixed; final scan = 0 |
| Preamble byte-identical to d13 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim |
| Visual review (rendered PNG @90dpi) | **PASS** | locked [I2], [K1] RAG skeleton, [K4] vector-DB code, [N] table, [Q] bridge — no overflow/clipping/collisions |
| Aux/PNG auto-cleaned | **PASS** | `.aux/.log/.nav/.out/.snm/.toc/.vrb` + `_r14/` removed; only `.tex`+`.pdf` remain |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Numbers (→ P14 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I2]** | RAG prompt `k·T_chunk + T_instr` (k=3, 200, 100) | **700 token; 700/128000 ≈ 0.55% < 1%; docs=3** | **P14 first assert** (course_map lock) |
| [J2] | prompt at k=5 | `5×200+100 = 1100` (≈0.86%) | task |
| [I1] | knowledge cutoff (LLM 2023, ask 2025) | correct-prob ≈ 0 vs RAG grounded | qualitative |
| [I3] | cosine `(1,1,1,0)·(1,1,0,1)` | `2/3 ≈ 0.667` (reuses L3) | semantic search |
| [J3] | cosine `(1,1,1,0)·(1,0,0,0)` | `1/√3 ≈ 0.577` | task |
| [I4] | 10000 chunks, brute vs ANN | `O(n)` vs `log₂10000 ≈ 13` (~770× faster) | vector DB |
| [J4] | ANN steps for 10⁶ vectors | `log₂(10⁶) ≈ 20` | task |

Only the RAG prompt-token count ([I2]) is locked by course_map; all other [I]/[J] numbers are arithmetic / `log`-table / cosine computable. [I3] deliberately reuses L3's `2/3` cosine.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L13 (fine-tune tunes, doesn't add facts) + P13; problem = hallucination + stale knowledge |
| LLM va RAG | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | open-book exam analogy; RAG = retrieve+generate; RAG vs fine-tune |
| RAG jarayoni | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | prompt budget; **locked [I2]**; lost-in-the-middle pitfall |
| Embeddinglar | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | semantic search, cosine (reuses L3), top-k; chunking pitfall |
| Vektor DB | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | Pinecone/Weaviate/FAISS; ANN/HNSW; upsert/query |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1 chunking][S2 HNSW] |

---

## Content Continuity

- **[B] recap** bridges L13: fine-tuning tunes weights but can't cheaply add new/external facts → motivates RAG.
- **[E] problem-first** shows LLM hallucination + knowledge cutoff *before* naming RAG.
- **Arc (L13 → L14)**: [F1]/[H1] state explicitly — fine-tuning changes **weights** (knowledge tuning); RAG adds facts to the **prompt** (knowledge retrieval). [N] table contrasts them (xulq/uslub→fine-tune; fakt/bilim→RAG).
- **Embedding reuse**: [F3]/[G3]/[I3] reuse L3/L6 embeddings + the L3 cosine `2/3` for semantic search.
- **[M] Uzbek**: lex.uz legal documents — hallucination-prone high-stakes domain; RAG grounds answers and cites sources; multilingual embedding quality (WordPiece suffix fragmentation, ties to L13 [M]).
- **[O] seminal**: Lewis et al. (2020) RAG + discussion (chunking + embedding choice for Uzbek legal text).
- **[Q] bridge**: TikZ pipeline knowledge base → chunk+embed → FAISS vector DB → retrieve top-k → m14 RAGEngine; first assert = the locked [I2] token budget.

---

## Compile Notes

- **Clean compile** — 0 errors, 0 Overfull >10pt; listing lines kept short (RAG/embedding/vector-DB code) per the L13 [K3]/[K4] lesson.
- **Cyrillic pre-compile scan caught 8 homoglyphs** (the most of any deck so far): `modelга` ×2, `manbани`,
  `manbaга`, `uzunlikка`, `uzunlikдan`, `ozгina`, `tezlikка` — all Latin-looking `г/к/д/н+и` mixed into
  Uzbek suffixes. All fixed; final scan = 0. (The §6 warning that this is the most stubborn defect held true.)
- Percent/fraction handled correctly: `<1\%` (escaped) in prose; `700/128000` and `\frac{700}{128000}` in math mode. No bare `%`/`_`/`^`.

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (LLM limits/RAG; RAG process; vector embeddings in search; vector DBs Pinecone/Weaviate),
seminal paper (Lewis 2020), uzbek_angle (RAG + lex.uz legal), and the locked hand_example (RAG prompt token
count 700, <1%) all match Day 14 `lecture_official_no: 14`. `gpu_required: true` pertains to the paired
practice (P14); the lecture is theory — [K] shows RAG / cosine-search / vector-DB code that is read, not executed.

---

## Pending

- **Overleaf human review** of prose quality — per `uzbek-course-style` (human gate).
- **P14** (Day 15 — m14 RAGEngine) consumes [I2] (prompt=700 token, docs=3) as its first assert.
- Commit pushed to origin/rtm per task instruction.
