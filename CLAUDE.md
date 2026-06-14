# NLP Professional Development Course — Material Production Project

You are building the complete teaching materials for a 4-week intensive NLP
course (15-iyun – 10-iyul 2026) for professional-development participants.
Read this file fully before any task. The skills in `.claude/skills/` are
BINDING — consult the relevant skill before producing each artifact type.

## Non-negotiable facts

- **Schedule**: Mon/Tue/Thu/Fri, 16 teaching days. Each day: practice session
  09:30–10:50 (I.R. Atadjanov) and lecture 11:00–12:20 (A.A. Abdulali).
  Wednesdays are project-milestone days (asynchronous; brief + feedback).
- **Day 1 is an orientation session** (tools, accounts, resources, environment
  verification) — not a content lecture. From Day 2 onward, each lecture is
  paired with the NEXT day's practice on the same topic (lecture precedes
  practice).
- **Audience term**: participants are called **"tinglovchi"** (plural:
  "tinglovchilar") in all Uzbek materials, and **"participant"** in English
  text and code. NEVER use "professor", "professor-o'qituvchi", "talaba", or
  "student" to refer to the audience in any task, project text, slide,
  notebook, or comment. "o'qituvchi" is also forbidden as an audience
  reference; it is allowed ONLY inside linguistic example words/sentences
  (e.g., the morphology demo "o'qi-tuv-chi-lar-imiz-dan"). The course must
  read as field-neutral professional development.
- **Compute**: all practice materials target Kaggle Notebooks free tier.
  See skill `kaggle-hardware`. Never design an exercise that assumes hardware
  beyond a single 16 GB T4/P100.
- **Course arc (capstone)**: every tinglovchi builds ONE cumulative project —
  an Uzbek-language document assistant — across all 16 days. Each practice
  session contributes a module; Wednesday milestones integrate them; the final
  day deploys it. Week themes: (1) classical pipeline + representations,
  (2) embeddings + recurrent models, (3) attention + transformers,
  (4) fine-tuning, RAG, agents, deployment.

## Language rules

- Slide and notebook prose: **Uzbek (Latin script)**. Established English NLP
  terms appear in parentheses on first use: "so'z qopi (Bag of Words)".
- Code identifiers: English. Code comments addressed to tinglovchilar: Uzbek.
- Internal docs (course_map.yaml, QA reports, commit messages): English.
- **Capitalization and prose quality:** slide/heading/box titles use Uzbek
  sentence case (first word + proper nouns only — never Title Case); prose
  must read as natural human-written Uzbek. Follow skill `uzbek-course-style`
  § "Uzbek prose & capitalization style". Applies to ALL lectures, notebooks,
  and briefs from L2 onward. Reference example: `course/lectures/d01_nlp_asoslari.tex`.

## Repository layout (create and maintain exactly)

```
course/
├── course_map.yaml          # single source of truth: 16 days + 4 milestones
├── lectures/                # dNN_topic.tex + compiled dNN_topic.pdf
├── practices/               # dNN_topic.ipynb (+ dNN_checkpoints/ data)
├── milestones/              # wN_milestone.md briefs (Wednesdays)
├── capstone/                # project spec, module interface contracts, rubric
├── day1_orientation/        # setup checklist, account guide, env-check notebook
└── qa/                      # per-day QA reports
```

## Workflow discipline

1. **One day at a time.** Produce a complete day pair (lecture + practice),
   run all quality gates, write the QA report, then STOP and wait for
   explicit human approval before the next day. Never batch ahead without
   being asked.
2. **course_map.yaml first.** No artifact may be produced for a day whose
   row in course_map.yaml is not approved. If a content decision is not
   derivable from the map or skills, STOP and ask — do not invent.
3. **Quality gates (must pass before presenting any day):**
   - Lecture: `pdflatex` twice, zero errors; zero Overfull \hbox warnings
     > 10pt; slide count 22–28 excluding `\appendix`; all archetypes present
     (see skill `lecture-beamer`).
   - Notebook: valid JSON (`nbformat` validate); executes top-to-bottom with
     `OFFLINE_FALLBACK = True` (see skill `practice-notebook`); every assert
     passes; zero cells requiring > 16 GB VRAM.
   - Terminology: grep for forbidden audience terms returns nothing
     (see skill `uzbek-course-style`).
4. **Traceability.** Every practice assert that checks a numeric result must
   cite the lecture slide where that number is computed by hand
   (comment: `# Ma'ruza [I]-slayd bilan solishtiring`).
5. Git: commit per artifact, message format
   `dayNN: lecture|practice|qa — short description`.
