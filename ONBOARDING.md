# Coworker Onboarding — NLP Course Materials Repo

Fast start for a new contributor. Read this first, then follow the links for depth.

---

## 1. Clone

```bash
git clone <repo-url>
cd nlp_course_uz
pip install -r requirements.txt
nbstripout --install          # strips notebook outputs before every git add
```

Confirm nbstripout is wired up:
```bash
git config filter.nbstripout.clean   # should print: nbstripout
```

---

## 2. Project constitution

**`CLAUDE.md`** is the project constitution — Claude Code reads it at the start of every session. It defines the schedule, audience rules, repo layout, workflow discipline, and quality gates. Read it before touching anything.

**`.claude/skills/`** contains binding production rules Claude Code auto-reads before generating each artifact type:

| Skill | When it applies |
|-------|----------------|
| `lecture-beamer` | Creating or editing any `.tex` lecture deck |
| `practice-notebook` | Creating or editing any `.ipynb` practice notebook |
| `uzbek-course-style` | Any tinglovchi-facing text (slides, notebooks, briefs, asserts) |
| `kaggle-hardware` | Any code, model choice, or training config |

These are not suggestions — they are enforced conventions. Claude Code will read the relevant skill before producing each file. You should read them too before reviewing Claude's output.

---

## 3. How work flows — `PROMPTS.md`

`PROMPTS.md` contains the exact staged prompts for Claude Code, in order. The discipline is:

1. Open Claude Code in the repo root.
2. Paste the next stage's prompt **verbatim**.
3. Claude produces one day-pair (lecture + practice + QA report) and stops.
4. You review the output in Overleaf/Kaggle and read `course/qa/dNN_report.md`.
5. If all gates pass and you approve — move to the next stage.

**Never batch ahead.** One day at a time. Claude will not proceed without explicit approval.

---

## 4. Compile and run environments

| Artifact | Environment | Notes |
|----------|-------------|-------|
| `.tex` lecture decks | **Overleaf** (pdfLaTeX) | No local TeX install required or expected. Upload the `.tex` file; compile twice. |
| `.ipynb` practice notebooks | **Kaggle Notebooks** (CPU, free tier) | Weeks 1–2 are CPU-only. See `kaggle-hardware` skill for ceilings. |
| Capstone modules (`capstone/modules/`) | Either | Pure Python; run locally or in Kaggle. |

---

## 5. Branch and PR

Every contribution lives on its own branch:

```
dayNN/<short-description>
```

Open a PR into `main`. Attach `course/qa/dNN_report.md` — a PR without a passing QA report will not be merged. Full rules in [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## 6. What never goes in the repo

- Tokens, API keys, credentials (`kaggle.json`, `.env`, `*.token` — already gitignored)
- Raw corpora, large datasets, trained model weights → use Kaggle Datasets
- Notebook outputs → nbstripout handles this automatically

---

## Uzbek summary (tinglovchilar uchun emas — hamkasblar uchun)

Repozitoriyni klonlang, `requirements.txt` dan paketlarni o'rnating va
`nbstripout --install` ni ishga tushiring. `CLAUDE.md` loyiha qoidalarini,
`.claude/skills/` esa har bir artefakt turi uchun majburiy ko'rsatmalarni
o'z ichiga oladi. Ish jarayoni: `PROMPTS.md` dagi keyingi bosqich promptini
Claude Code ga joylashtiring → natijani Overleaf/Kaggle da ko'rib chiqing →
`qa/dNN_report.md` ni tasdiqlang → navbatdagi bosqichga o'ting.
Har kuni bittadan. Har bir o'zgarish alohida `dayNN/...` branchda.
