# QA Report: Day 16 Lecture (L16) — FINAL LECTURE

**Artifact**: `course/lectures/d16_mlops.tex` → `d16_mlops.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 16, `lecture_official_no: 16` — "NLP amaliyotida MLOps amaliyotlari"
**Paired practice**: NONE — `hand_example: null` (last lecture; no following P session)
**Recap target [B]**: L15 (agents) + this morning's P15 (capstone agent)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0; `Select-String "^!"` empty |
| Zero `Overfull \hbox` > 10pt | **PASS** | `Select-String "Overfull \hbox ([1-9][0-9]"` empty (0) |
| Slide count (L1–L15 parity, 4 full cycles) | **PASS** | **47 frames** (footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | H1 deploy stages, H2 Docker, H3 drift detection, H4 CI/CD automation |
| [M] Uzbek-language slide (mandatory) | **PASS** | "O'zbek NLP ekotizimi va ochiq muammolar" (UzBERT/Tahrirchi; labeled data, morphology, domain LLM) |
| Every formula has `\bunda{}` key | **PASS** | [G1] lifecycle, [G2] API/image, [G3] drift (Δ>τ), [G4] CI/CD; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] + appendix S2 all `[fragile]` (FastAPI/Dockerfile/yaml/curl) |
| **[I] hand_example — NONE (course_map null)** | **PASS (by design)** | No locked assert; [I1]/[I2]/[I3]/[I4] are illustrative (load-once, image layers, drift Δ=0.15, pipeline timing) — **no `# Ma'ruza` traceability comment** (no following P) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi`; labels `ijobiy`/`salbiy` |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | pre-compile Cyrillic scan = 0 (clean from the start — no homoglyph slipped in, unlike L14/L15) |
| Preamble byte-identical to d15 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim |
| Visual review (rendered PNG @90dpi) | **PASS** | [K2] Dockerfile, [I3] drift, [K4] CI yaml, [Q] defense bridge, [S2] curl smoke — JSON/`{}` braces render correctly; no overflow/clipping/collisions |
| Aux/PNG auto-cleaned | **PASS** | `.aux/.log/.nav/.out/.snm/.toc/.vrb` + `_r16/` removed; only `.tex`+`.pdf` remain |

**Overall: ALL GATES PASS**

---

## Hand-Worked Numbers (ILLUSTRATIVE — none locked)

L16 has `hand_example: null` in course_map (no following practice), so **no [I] feeds a P assert**. The [I]
slides are illustrative only (no traceability comments):

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| [I1] | load-once vs per-request | 2020 ms vs 20 ms (~100×) | deploy efficiency |
| [J1] | throughput (1 proc) | 1000/20 = 50 req/s | task |
| [I2] | Docker image layers | 120 + 800 + 400 ≈ 1320 MB | image structure |
| [I3] | drift `Δ = F1₀ − F1` | 0.85−0.70 = 0.15 > 0.10 = τ → retrain | monitoring |
| [J3] | drift Δ | 0.88−0.82 = 0.06 < 0.10 → ok | task |
| [I4] | pipeline timing | 30+90+20 ≈ 140 s | CI/CD |

All computable mentally; none are course_map-locked (no P15→P16 assert from L16).

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L15 + P15 (capstone agent); problem = "menda ishlaydi" ≠ production |
| Deploy | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | train→package→serve→monitor; training-serving skew pitfall |
| API va Docker | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | FastAPI + Dockerfile; image layers/cache; secrets pitfall |
| Monitoring | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | drift Δ>τ; model registry/versioning; rollback pitfall |
| CI/CD | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | commit→test→build→deploy; test-gate pitfall |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1 registry/A-B][S2 smoke test] |

---

## Content Continuity & Course Finale

- **[B] recap** bridges L15: the capstone agent works on the notebook → how to make it a reliable service?
- **[E] problem-first** shows "works on my machine" + silent drift *before* naming MLOps.
- **M4 P16 link**: the FastAPI/Docker flipped session (M4) practised this; L16 **formalizes** it. [K1]/[K2] mirror the P16 FastAPI/Dockerfile.
- **[M] Uzbek**: Uzbek NLP ecosystem (UzBERT, Tahrirchi), open problems (labeled data, morphology, domain LLMs), contribution directions.
- **[O] seminal**: Sculley et al. (2015) Hidden Technical Debt + discussion (which "hidden debt" is worst for Uzbek services).
- **[Q] = capstone defense** (NOT a next practice): TikZ m15 agent → SentimentAPI → Docker deploy; "tayyorlab keling" defense demo checklist. This is the **course finale**.
- **[R]** closes the whole course (16 days, classical NLP → agents) with congratulations + a call to contribute to Uzbek NLP.

---

## Compile Notes

- **Clean first compile** — 0 errors, 0 Overfull >10pt; code-heavy deck (FastAPI/Dockerfile/yaml/curl), all
  listings `[fragile]` with short lines.
- **Cyrillic pre-compile scan = 0** — no homoglyph slipped in this time (unlike L14's 8 and L15's 5). The
  recurring-defect discipline held.
- **`{}` escaping correct**: JSON in prose (`{sentiment, confidence}`, request/response) escaped or in
  `\texttt{}`; code-block braces (curl, Docker CMD list) left literal inside `lstlisting`; PNG review confirms
  braces render.
- **No locked [I]** — by course_map design (`hand_example: null`); no fabricated "locked" number; no
  `# Ma'ruza L16 [I]-slayd` comment anywhere; [Q] points to the defense, not a next P.

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (deploy stages; API+Docker; monitoring+versioning; CI/CD), seminal paper (Sculley
2015), uzbek_angle (Uzbek NLP ecosystem), and `hand_example: null` (no following practice → no locked assert)
all match Day 16 `lecture_official_no: 16`. `gpu_required: false` — lecture is theory; [K] shows FastAPI/
Dockerfile/CI-yaml/curl code that is read, not executed. As the last lecture, [Q] is adapted to the capstone
defense (the lecture-beamer "bridge to tomorrow's practice" archetype, repurposed since no P16 follows L16).

---

## Pending

- **Overleaf human review** of prose quality — per `uzbek-course-style` (human gate).
- **No paired practice** — L16 is the final lecture.
- Remaining course artifact: **w4 (M4) milestone** (P16 FastAPI/Docker fully-worked, knowledge test, agent
  scaffold) — after which the course is fully complete.
