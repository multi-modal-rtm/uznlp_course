# Contributing to the NLP Course Materials

This repository uses a strict one-day-at-a-time review gate. Read this document
fully before pushing anything.

## Branch Naming

Every contribution lives on its own branch named after the day it targets:

```
dayNN/<short-description>
```

Examples:
- `day02/preprocessing-bow-tfidf` — lecture + practice pair for Day 2
- `day03/qa-fixes` — corrections after QA review of Day 3
- `day01/infra-gitignore` — infrastructure work referencing the closest session

Never push directly to `main`. The `main` branch contains only approved,
gate-passing artifacts.

## Pull Request Flow

1. Open a PR from your `dayNN/...` branch into `main`.
2. Complete the PR template checklist (see `.github/pull_request_template.md`).
3. **Attach `course/qa/dNN_report.md`** as the approval artifact — a PR without a
   passing QA report will not be merged.
4. Assign the relevant course maintainer for review.
5. Address review comments on the same branch; do not open new PRs for fixes.
6. Merge only after explicit written approval.

## Setting Up nbstripout (Required Before First Commit)

Notebook outputs must never be committed. Install the `nbstripout` git filter
once per local clone:

```bash
pip install -r requirements.txt
nbstripout --install
```

This registers a git clean filter that automatically strips outputs when you
`git add` any `.ipynb` file. Verify the filter is active:

```bash
git config filter.nbstripout.clean
# Expected output: nbstripout
```

If nbstripout is not installed, any `git add` of a notebook with non-empty
outputs will include those outputs in the commit — which may leak data or
inflate repo size.

## Secrets Rules

**Never commit tokens, API keys, or credentials** — not even temporarily.

- `kaggle.json`, `.env`, and `*.token` are listed in `.gitignore` and must stay
  there.
- In notebooks, read secrets from environment variables or Kaggle Secrets only:

  ```python
  import os
  # Mahalliy muhitda: .env faylidan o'qing (repo-ga kiritmang)
  # HF_TOKEN=hf_xxx — .env (gitignored) faylida saqlang
  token = os.environ.get('HF_TOKEN', '')

  # Kaggle Notebooks-da:
  # from kaggle_secrets import UserSecretsClient
  # token = UserSecretsClient().get_secret('HF_TOKEN')
  ```

- For verification, print at most the last 4 characters:
  `print(f"Token yuklandi: hf_...{token[-4:]}")`. Never print the full value.
- If you accidentally commit a secret, **rotate the credential immediately** —
  deleting the file in a follow-up commit is not sufficient; the secret remains
  in git history. Contact the course maintainer to scrub history.

## Large Files — Never Commit

Raw corpora, cleaned datasets, trained model weights, and binary vector stores
live on **Kaggle Datasets**, not in this repository. See
`.claude/skills/kaggle-hardware/SKILL.md` for dataset attachment instructions.

Gitignored patterns: `corpus/raw/`, `corpus/clean/`, `datasets/`, `*.bin`,
`*.kv`, `*.safetensors`, `*.pkl`, `*.pt`, `*.ckpt`.

To share a dataset with a collaborator, provide the Kaggle Dataset URL — do not
include data files in the PR.

## Terminology Check (Run Before Every Commit)

Audience term in Uzbek: **tinglovchi / tinglovchilar**
Audience term in English: **participant(s)**

Forbidden in any artifact (slides, notebooks, scripts, docs):
`professor`, `talaba`, `student`, `o'qituvchi` (as audience label)

Run before staging new or edited files:

```bash
grep -rinE "professor|talaba|student|o'qituvchi" <file>
```

Every match must be a linguistic example (`o'qituvchi` inside a morphology demo
word only) — any other match is a terminology defect that must be fixed before
the PR is opened.

## Commit Message Format

```
dayNN: lecture|practice|qa|infra — short description
```

Examples:
- `day02: lecture — preprocessing BoW TF-IDF Boadilla deck`
- `day02: practice — P1 preprocessing pipeline notebook`
- `day02: qa — L2 P1 report all gates PASS`
- `day01: infra — gitignore secrets LaTeX artifacts large data`
