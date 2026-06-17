# QA Report: Day 16 Practice (P15) — CAPSTONE FINALE

**Artifacts**: `course/practices/d16_p15_agent.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m15_langchain_agent.py` (FINAL integration module — defense demo; `consumed_by: []`)
**Bundled data**: none (small inline corpora: 6 sentiment rows + 3 KB chunks)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L15 — Sun'iy intellekt agentlarini yaratish (ReAct)
**Next**: L16 (Day 16 lecture — MLOps) + w4 milestone

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `nbformat.validate` OK; nbformat 4.5; 26 cells (12 code, 14 md); all `id` |
| JSON valid — solutions | **PASS** | `nbformat.validate` OK; nbformat 4.5; 26 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True, CPU)** | **PASS (local)** | SOLUTIONS exec'd on Python 3.13.14 — **14/14 asserts passed**, zero exceptions. Offline rule-based keyword router; tools = real m13/m14/m12 + stubs (m04/m10). |
| Student stub cells compile | **PASS** | All 12 code cells `compile()` clean |
| **Locked assert (ReAct tool selection)** | **PASS** | §4A `agent.route("Uzum Market ilovasi yaxshimi?")` → `tool=="sentiment_classify"`, `confidence>0.7` (0.8) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L15 [I1]-slayd` (§4A) |
| Every blanked region has paired assert | **PASS** | 2 blanked cells (§4B agent build, §4C run) → paired asserts; §4D mustaqil + assert |
| m15 contract conformance | **PASS** | `run(user_message)->str` exact (contracts.py); `route()`/`last_trace` as documented extra (assertability) |
| **run() structural** | **PASS** | returns non-empty Uzbek str; `last_trace` has Thought/Action + Observation steps |
| 5-tool wiring | **PASS** | sentiment_classify→m13, retrieve_docs→m14, summarize_text→m12, spell_correct→m04(stub), extract_entities→m10(stub) |
| No GPU / VRAM | **PASS** | CPU-only; rule-based router; VRAM peak 0 GB |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, rebuilds agent) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi`; labels `ijobiy`/`salbiy` |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | pre-build scan caught 5 `agentга` homoglyphs in builder md, fixed & rebuilt; final scan = 0 |

**Overall: ALL GATES PASS**

---

## langchain/LLM absent — rule-based router fallback verified

- `langchain` absent locally (guard catches any exception → `HAS_LANGCHAIN=False`); no LLM API.
- Forced offline (`USE_LANGCHAIN=False`): m15 uses a **rule-based o'zbekcha keyword router** that models the
  ReAct "Thought" — picks the correct tool by keywords, calls it, assembles an Uzbek answer (`last_trace`
  records Thought/Action/Observation).
- The 5 tools are dependency-injected. `make_tools()` defensively loads **real** m13 (LogReg), m14 (TF-IDF),
  m12 (extractive) and uses lightweight stubs for m04/m10 (and falls back to stubs if any module import fails
  — e.g., a transient torch DLL error).

| Path | planning | tools |
|---|---|---|
| Kaggle (USE_LANGCHAIN=True, LLM) | `create_react_agent` + `AgentExecutor` (LLM) | LangChain `Tool` wrappers |
| offline (forced, local) | rule-based keyword router | real m13/m14/m12 + m04/m10 stubs |

`route()` is pure keyword logic (no tool/LLM needed) → the locked [I1] assert is path-independent and deterministic.

---

## Locked / Verified Numbers

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I1]** | ReAct tool selection for "Uzum Market ilovasi yaxshimi?" | **tool=sentiment_classify; confidence=0.8 > 0.7** | **P15 first assert** (course_map lock) |

§4A reproduces the lecture's ReAct tool selection — **P15's first assert** — matching course_map Day 16
paired-lecture L15 `hand_example` (`# Ma'ruza L15 [I1]-slayd`). All 5 routes verified (sentiment/summarize/
entities/spell/retrieve).

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — capstone-finale framing, objectives from L15 [C], session_role disclosure |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, HAS_LANGCHAIN, USE_LLM, module path |
| §2 Yaxlit natija | 2 | MD+Code | OK — make_tools (5 tools) + agent.run demo + trace |
| §3 PRIMM periferiya | 3 | Mixed | OK — LangChain (Kaggle, commented) + rule-based routes; Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked route) + 4B wire+route + 4C run + 4D complex, each + assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m15 contract test (run), git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — route sweep; capstone congratulations; exit ticket |

Total: 26 cells (12 code, 14 markdown). Blanked core cells: 2 (§4B agent build, §4C run), each paired with an assert.

---

## Module Conformance (contracts.py)

```
m15 DocumentAssistantAgent (capstone finale, consumed_by: []):           provides:
  run(user_message: str) -> str                                          ✓  (Uzbek answer; full ReAct loop)
  route(message) -> dict {tool, confidence, args}                        ✓  (extra — ReAct "Thought", assertable)
  last_trace: list[dict]                                                 ✓  (extra — Thought/Action/Observation)
```
Contract requires only `run()`; `route()`/`last_trace` are documented extras for observability/assertability.
Tools (sentiment_classify/retrieve_docs/summarize_text/spell_correct/extract_entities) injected via `tools` dict.

---

## Deviation from course_map.yaml

course_map Day 16 `corpus_subset: uz_kb`; `session_role: wiring_and_polish` (M4 Task C scaffold + Day 15
partial wiring). No separate scaffold artifact exists yet (M4/w4 not produced), so m15 wires the existing
capstone modules directly. Local run is **CPU-only** with a **rule-based keyword router** (langchain/LLM
Kaggle-only) and small inline corpora (no bundled checkpoint file needed). m04/m10 wired as lightweight
stubs (real modules optional); m13/m14/m12 wired real (offline fallbacks). Answer quality is demo-grade
(rule-based), stated honestly. The keyword router can mis-route on substring collisions (e.g., "nom" inside
"shartnomasi") — disambiguated demo queries are used; real ReAct/LLM path handles nuance.

---

## Pending

- Full Kaggle kernel run with LangChain `create_react_agent` + LLM (real ReAct multi-step orchestration of
  all 5 tools) — confirmed when notebooks are published as a Kaggle Dataset (Day 16, 10-iyul-2026).
- **L16** (Day 16 lecture — MLOps) and **w4** milestone are the next artifacts.
- m15 is the capstone finale: used directly in the defense demo (`consumed_by: []`).
