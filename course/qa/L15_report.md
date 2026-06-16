# QA Report: Day 15 Lecture (L15)

**Artifact**: `course/lectures/d15_agentlar.tex` → `d15_agentlar.pdf`
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local MiKTeX compile + visual review)
**course_map**: Day 15, `lecture_official_no: 15` — "Sun'iy intellekt agentlarini yaratish"
**Paired practice**: P15 (Day 16 — m15 DocumentAssistantAgent, LangChain ReAct, 5 tools)
**Recap target [B]**: L14 (RAG) + this morning's P14

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| `pdflatex` ×2 — zero errors | **PASS** | Exit 0; `Select-String "^!"` empty |
| Zero `Overfull \hbox` > 10pt | **PASS** | `Select-String "Overfull \hbox ([1-9][0-9]"` empty (0) |
| Slide count (L1–L14 parity, 4 full cycles) | **PASS** | **47 frames** (footer `/47`). NOT reduced to 22–28. |
| All archetypes [A]–[S] present | **PASS** | [A][B][C][D][E] + 4× [F][G][H][I][J][K][L] + [M][N][O][P][Q][R] + [S] appendix |
| **[H1]–[H4] all present** | **PASS** | H1 reason+act, H2 function-calling, H3 model-as-API, H4 LangChain framework |
| [M] Uzbek-language slide (mandatory) | **PASS** | "Agent fikrlashi va tool chaqiruvi" (ReAct Thought in Uzbek; tool selection reliability) |
| Every formula has `\bunda{}` key | **PASS** | [G1] ReAct, [G2] tool/selection, [G3] API contract, [G4] AgentExecutor; appendix S1/S2 |
| `[fragile]` on all `lstlisting` frames | **PASS** | [K1][K2][K3][K4] + appendix S1 all `[fragile]` |
| **Locked [I1] hand_example** | **PASS** | ReAct: correct_tool=`sentiment_classify`, Observation `{sentiment: ijobiy, confidence: 0.82}`, conf 0.82>0.7 — verbatim from course_map |
| Traceability comment → P15 | **PASS** | [I1] + [Q] carry `# Ma'ruza L15 [I1]-slayd`; [Q] shows `assert tool == "sentiment_classify"` / `assert conf > 0.7` |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi`; labels `ijobiy`/`salbiy` |
| ASCII apostrophe / no U+2019 / **no Cyrillic** / no BOM | **PASS** | pre-compile scan caught **5** homoglyphs (chiqarilгuncha, endpointга, agentга ×3), all fixed; final scan = 0 |
| Preamble byte-identical to d14 | **PASS** | colors, Boadilla footer, lstset, tcbset, tikz styles copied verbatim |
| Visual review (rendered PNG @90dpi) | **PASS** | locked [I1] ReAct trace, [I3] API JSON, [K4] LangChain code, [Q] bridge — braces render correctly; no overflow/clipping/collisions |
| Aux/PNG auto-cleaned | **PASS** | `.aux/.log/.nav/.out/.snm/.toc/.vrb` + `_r15/` removed; only `.tex`+`.pdf` remain |

**Overall: ALL GATES PASS**

---

## Locked / Hand-Worked Numbers (→ P15 asserts)

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I1]** | ReAct tool selection (query "Uzum Market ilovasi yaxshimi?") | **correct_tool=sentiment_classify; conf=0.82 > 0.7** | **P15 first assert** (course_map lock) |
| [J1] | tool for "qisqacha bayon qil" | `summarize` | task |
| [I2] | tool by description ("shahar nomlarini top") | `extract_entities` | function calling |
| [J2] | tool for "imlo xatosi" | `spell_correct` | task |
| [I3] | API response for "mahsulot zo'r" | `{sentiment: ijobiy, confidence: 0.91}` | model-as-API |
| [J3] | API response for "juda yomon xizmat" | `{sentiment: salbiy, confidence: 0.88}` | task |
| [I4] | modules → tools | 5 (m13/m14/m12/m04/m10) | LangChain wiring |
| [J4] | add m11 translate | 6 tools | task |

Only the ReAct tool-selection ([I1]) is locked by course_map; all other [I]/[J] are qualitative tool-routing / JSON-contract examples.

---

## Structure (6 sections, 4 full cycles)

| Section (short / full) | Archetypes | Notes |
|------------------------|-----------|-------|
| Kirish | [A][B][C][D][E] | recap L14 (RAG = one tool) + P14; problem = LLM can't act / plan multi-step |
| ReAct | Cycle 1: [F1][G1][H1][I1][J1][K1][L1] | Thought→Action→Observation; **locked [I1]**; infinite-loop pitfall |
| Tools | Cycle 2: [F2][G2][H2][I2][J2][K2][L2] | function calling; tool name/description/schema; vague-description pitfall |
| Arxitektura | Cycle 3: [F3][G3][H3][I3][J3][K3][L3] | NLP model as FastAPI; request/response JSON contract |
| LangChain | Cycle 4: [F4][G4][H4][I4][J4][K4][L4] | Tool/AgentExecutor; wiring 5 capstone modules |
| Xulosa | [M][N][O][P][Q][R] | + appendix [S1 ReAct prompt template][S2 agent memory] |

---

## Content Continuity

- **[B] recap** bridges L14: RAG performed one action (retrieve); multi-step tasks need planning + multiple tools → agent.
- **[E] problem-first** shows LLM can't act on the world / can't plan multi-step *before* naming agents.
- **Arc (L14 → L15)**: [F1]/[N] state explicitly — RAG = one tool (retrieve); agent plans and calls **many** tools. The capstone modules m13/m14/m12/m04/m10 become the agent's tools.
- **[M] Uzbek**: ReAct "Thought" reliability in Uzbek; tool-selection on Uzbek queries; mitigation (clear tool descriptions, English internal reasoning + Uzbek final answer).
- **[O] seminal**: Yao et al. (2022) ReAct + discussion (testing tool selection on Uzbek queries).
- **[Q] bridge**: TikZ pipeline m13/m14/m12 → m15 Agent (ReAct); first assert = the locked [I1] tool selection.

---

## Compile Notes

- **Clean compile** — 0 errors, 0 Overfull >10pt; listing lines kept short (langchain/JSON) per the L13/L14 lesson.
- **Cyrillic pre-compile scan caught 5 homoglyphs**: `chiqarilгuncha`, `endpointга`, `agentга` ×3 — again
  Latin-looking `г` inside Uzbek suffixes (-ga/-guncha). All fixed; final scan = 0. (Recurring defect, expected and caught.)
- **`{}` escaping handled correctly**: JSON/dict shown in prose (`{sentiment: ijobiy, confidence: 0.82}`,
  API request/response) uses escaped `\{...\}` or `\texttt{}`; rendered braces verified in PNG review. Code-block
  JSON inside `[fragile]` `lstlisting` (tool schema, ReAct template) left literal.
- Percent/math handled in math mode; `\ding`/`\psmallmatrix` not used.

---

## Deviation from course_map.yaml

None. Topic, 4 subitems (ReAct; function calling/tools; agent architecture + model-as-API; LangChain),
seminal paper (Yao 2022 ReAct), uzbek_angle (Uzbek reasoning/tool selection), and the locked hand_example
(ReAct tool selection: sentiment_classify, conf 0.82) all match Day 15 `lecture_official_no: 15`.
`gpu_required: false` — lecture is theory; [K] shows ReAct/tool/FastAPI/LangChain code that is read, not executed.

---

## Pending

- **Overleaf human review** of prose quality — per `uzbek-course-style` (human gate).
- **P15** (Day 16 — m15 DocumentAssistantAgent) consumes [I1] (correct_tool=sentiment_classify, conf>0.7) as its first assert.
- ⚠️ Unpushed (before this commit): P14 (4 commits) + this L15 commit — origin/rtm at `bebcc12`; push when instructed.
