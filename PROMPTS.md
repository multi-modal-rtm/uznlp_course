# Claude Code Prompt Sequence — NLP Course Production

Run stages strictly in order. After each stage, review the outputs and the
QA report yourself before continuing. Paste each prompt verbatim.

---

## STAGE 0 — Bootstrap and course map (run first)

```
Read CLAUDE.md and every skill in .claude/skills/ completely before doing
anything. Then:

1. Create the repository layout exactly as specified in CLAUDE.md and
   initialize git.

2. Produce course/course_map.yaml — the single source of truth. One entry
   per teaching day (16 days, 15-iyun – 10-iyul 2026, Mon/Tue/Thu/Fri) plus
   4 Wednesday milestone entries. Schema per day:
     day, date, week, lecture_topic (Uzbek), practice_topic (Uzbek),
     periphery (list: parts given as complete PRIMM code),
     core (list: parts taught with faded scaffolding),
     capstone_module (the module this day adds, with file name),
     hand_example (the tiny worked example shared by lecture slide [I] and
       the practice's first assert — describe it and the expected numeric
       result),
     gpu_required (true/false — follow skill kaggle-hardware: weeks 1–2
       false),
     seminal_paper (for lecture archetype [O]),
     uzbek_angle (the day's mandatory Uzbek-language consideration).
   Day 1 is the orientation session (no lecture archetypes; see CLAUDE.md).
   From Day 2 on, day N's lecture topic equals day N+1's practice topic.
   Week themes and the cumulative capstone arc are in CLAUDE.md.

3. Sanity-check the map: every capstone_module consumed later must be
   produced earlier; every hand_example is mentally computable; flag any
   day where the 80-minute budget looks unrealistic and propose a fix.

4. STOP. Print the full course_map.yaml and your sanity-check notes for my
   approval. Do not produce any lecture or notebook yet.
```

---

## STAGE 1 — Capstone spec + Day 1 orientation (after map approval)

```
course_map.yaml is approved. Read it plus all skills again if context was
compacted. Now produce, in this order:

1. capstone/SPEC.md — the cumulative project specification (Uzbek,
   tinglovchi-facing): what the Uzbek document assistant does when finished,
   the module list mapped day-by-day from course_map.yaml, the interface
   contract for each module (function signatures, input/output types) in
   capstone/contracts.py, the 4 Wednesday milestone definitions, and the
   final-day defense rubric (criteria + levels).

2. day1_orientation/ materials:
   a. d01_orientatsiya.ipynb — environment verification notebook following
      skill practice-notebook structure where applicable: account checklist
      cells (Kaggle + phone verification for GPU, GitHub, Hugging Face
      token), a CPU smoke test, a one-cell GPU smoke test with quota
      warning, dataset-attach walkthrough, git configuration, and a final
      cell that prints "Muhit tayyor" only when every check passes.
   b. d01_kirish.tex — a SHORT deck (12–16 slides) from the lecture-beamer
      template: course goals, the capstone arc as a TikZ pipeline, weekly
      map, how each day works (lecture↔practice pairing), compute resources
      and quota etiquette, and the Day-2 preparation list. Skip archetypes
      [F]–[O]; keep [A],[C],[D],[Q],[R].
   c. day1_orientation/HISOB_YARATISH.md — step-by-step account setup guide
      (Uzbek) tinglovchilar complete BEFORE Day 1.

3. Run every quality gate from CLAUDE.md (compile checks, notebook
   execution with OFFLINE_FALLBACK, terminology grep). Write
   qa/day01_report.md. STOP and present the report.
```

---

## STAGE 2 — Day 2 gold-standard pair (after Stage 1 approval)

```
Produce the Day 2 materials as the GOLD STANDARD every later day will copy.
Day 2 per course_map.yaml: practice = end-to-end "O'zbek matni tahlilchisi"
(web scraping + PDF extraction as periphery via PRIMM; preprocessing
pipeline + BoW + TF-IDF as core via faded scaffolding); the paired lecture
(delivered Day 2, 11:00) prepares Day 3's practice topic per the map.

1. lectures/d02_*.tex from .claude/skills/lecture-beamer/assets/
   00_marquza_shabloni.tex — every archetype [A]–[S], hand_example numbers
   from course_map.yaml.
2. practices/d02_*.ipynb + d02_*_SOLUTIONS.ipynb following skill
   practice-notebook exactly: section timings in the header, PRIMM prompts
   on every periphery chunk, the three-stage fade on the core,
   checkpoint cells with bundled data in practices/d02_checkpoints/,
   capstone integration cell writing the day's module against
   capstone/contracts.py, investigation question, exit ticket.
3. The first core assert must check the exact numbers from the Day 2
   lecture's [I] slide (traceability comment per CLAUDE.md).
4. Run all quality gates including executing BOTH notebooks offline.
   Write qa/day02_report.md listing: gate results, slide count, total
   estimated runtime, VRAM peak (must be 0 — CPU day), and any judgment
   calls you made. STOP and present.
```

---

## STAGE 3 — Complete Week 1 (after Day 2 approval)

```
Day 2 is approved as the gold standard. Produce Days 3 and 4 (lecture +
practice + solutions + checkpoints + QA report each), one day at a time,
stopping after each day's QA report for my approval before the next.
Match the gold standard's structure cell-for-cell and archetype-for-
archetype; only content changes. Then produce
milestones/w1_milestone.md: the Wednesday brief (Uzbek) — integration
tasks for capstone modules built in days 1–4, a self-check script
milestones/w1_check.py that asserts the integrated modules satisfy
capstone/contracts.py, and a short written-reflection prompt.
```

---

## REPEATING PATTERN — Weeks 2–4

Use the Stage 3 prompt as the pattern, substituting the week number and
days. GPU days additionally require in the QA report: measured peak VRAM,
single-cell max runtime, and confirmation of checkpointing per skill
kaggle-hardware. Before starting Week 4, ask for the consolidated review
prompt (provided separately after Week 3 review).

## FINAL AUDIT (after all 16 days)

```
Run a full-course consistency audit: (1) execute every notebook offline in
sequence, asserting each capstone module imports cleanly into the next
week's consumers; (2) re-run terminology grep across the repo; (3) verify
every lecture [Q] slide matches the NEXT day's practice header; (4) check
all 16 hand_examples appear in both their lecture and their notebook with
identical numbers; (5) produce qa/final_audit.md with a pass/fail matrix
and a list of every judgment call awaiting human review.
```
