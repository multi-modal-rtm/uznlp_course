# NLP Professional Development Course — Uzbek (2026)

Intensive 4-week Natural Language Processing course delivered in Uzbek (Latin script)
for professional-development participants. Runs **15 June – 10 July 2026**,
16 teaching days (Mon/Tue/Thu/Fri).

Each participant builds one cumulative project — an **Uzbek-language document assistant**
— across all 16 sessions. All practices run on **Kaggle Notebooks free tier**
(T4/P100; no local GPU required).

## Schedule at a Glance

| Week | Theme | Days |
|------|-------|------|
| 1 | Classical pipeline + sparse representations | Days 1–4 |
| 2 | Embeddings + recurrent models | Days 5–8 |
| 3 | Attention + transformers | Days 9–12 |
| 4 | Fine-tuning, RAG, agents, deployment | Days 13–16 |

Practice sessions: 09:30–10:50 (I.R. Atadjanov)
Lectures: 11:00–12:20 (A.A. Abdulali)
Wednesdays: project milestone review (asynchronous)

## Repository Layout

```
course/
├── course_map.yaml          # source of truth: 16 days + 4 milestones
├── lectures/                # dNN_topic.tex + compiled dNN_topic.pdf
├── practices/               # dNN_topic.ipynb (+ dNN_checkpoints/ data)
├── milestones/              # wN_milestone.md briefs (Wednesdays)
├── capstone/                # project spec, module contracts, rubric
├── day1_orientation/        # setup checklist, account guide, env-check notebook
└── qa/                      # per-day QA reports (approval gates)
.claude/
└── skills/                  # BINDING production rules — read before any artifact
```

## Production Rules — `.claude/skills/` Are Binding

All teaching materials are produced following the skills in `.claude/skills/`.
Every artifact must pass the gates defined in the relevant skill before it is presented.

| Skill | Governs |
|-------|---------|
| `lecture-beamer` | Lecture deck archetypes [A]–[S], compile gates, slide count |
| `practice-notebook` | PRIMM structure, offline execution, capstone integration |
| `kaggle-hardware` | Hardware ceilings, batch size limits, data size caps |
| `uzbek-course-style` | Terminology, language conventions, forbidden audience terms |

## PROMPTS.md Staged Workflow

`PROMPTS.md` drives multi-step production via staged prompts. Each prompt produces
one artifact, runs quality gates, writes `course/qa/dNN_report.md`, and waits for
human approval. Never batch ahead. The cycle is:

**Produce → QA gates → `qa/dNN_report.md` → Human approves → Next day**

## Core Discipline

- **One day at a time.** Produce lecture + practice, run all gates, write the QA
  report, then stop and wait for explicit approval.
- **`course_map.yaml` is the source of truth.** No artifact may be created for a
  day whose row in the map is not approved.
- **`qa/dNN_report.md` is the approval gate.** A day pair is complete only when its
  QA report is present and all gates show PASS.
- **No raw corpus or large model files in git.** Datasets live on Kaggle Datasets
  (see `kaggle-hardware` skill).
- **Audience terminology.** Participants are called `tinglovchi`/`tinglovchilar` in
  Uzbek materials and `participant(s)` in English. Terms `student`, `talaba`,
  `professor`, and `o'qituvchi` (as audience label) are forbidden in all artifacts.

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, PR flow, and secrets rules.

---

## O'zbek qisqacha bayoni

Ushbu repozitoriy **NLP bo'yicha kasbiy malaka oshirish kursi** materiallarini
o'z ichiga oladi (15-iyun — 10-iyul 2026, 16 o'quv kuni).

Har bir tinglovchi 16 kun davomida yagona kapstone loyiha — **o'zbek tili hujjat
assistenti** — quradi. Barcha amaliyotlar Kaggle Notebooks bepul rejimida
(T4/P100 GPU) bajariladi; mahalliy GPU talab qilinmaydi.

**Kurs tuzilishi (hafta bo'yicha):**

| Hafta | Mavzu |
|-------|-------|
| 1 | Klassik pipeline + siyrak vektorlar (BoW, TF-IDF) |
| 2 | Embeddinglar + rekurrent modellar (Word2Vec, LSTM) |
| 3 | Diqqat mexanizmi + transformerlar (BERT) |
| 4 | Fine-tuning, RAG, agentlar, deployment |

**Asosiy qoidalar:**
- Materiallar kunlik usulda ishlab chiqiladi: ma'ruza → amaliyot → sifat tekshiruvi → tasdiqlash → keyingi kun.
- `.claude/skills/` papkasidagi ko'rsatmalar **MAJBURIY** — har bir artefakt ushbu qoidalarga mos bo'lishi shart.
- Katta ma'lumotlar (corpus, model og'irliklari) Kaggle Datasets-da saqlanadi; git repozitoriyiga kiritilinmaydi.
- Tinglovchi terminologiyasi: `tinglovchi`/`tinglovchilar` (o'zbek tilida), `participant(s)` (ingliz tilida). `talaba`, `student`, `professor` so'zlari materiallarida taqiqlangan.
