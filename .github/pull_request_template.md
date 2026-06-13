## Day pair / artifact

**Day NN — topic name**

Artifacts in this PR:
- [ ] `course/lectures/dNN_topic.tex` (lecture)
- [ ] `course/practices/dNN_topic.ipynb` (practice)
- [ ] `course/qa/dNN_report.md` (QA report — required for merge)

---

## Quality gates

### Lecture (LaTeX/Beamer)

- [ ] `pdflatex` run **twice**, zero `^!` errors in log
- [ ] No `Overfull \hbox` > 10 pt in log
- [ ] Slide count 22–28 (body only, before `\appendix`)
- [ ] All archetypes present:
  [A] title · [B] recap · [C] objectives · [D] plan · [E] problem-first
  motivation · [F] intuition · [G] defbox+`\bunda` · [H] derivation ·
  [I] hand example · [J] your-turn warnbox→okbox · [K] code↔formula ·
  [L] pitfall · [M] Uzbek angle · [N] comparison table · [O] seminal paper ·
  [P] objectives checkmarked · [Q] bridge to practice · [R] references ·
  [S] appendix
- [ ] Terminology grep clean: `grep -rinE "professor|talaba|student|o'qituvchi" course/lectures/dNN_*.tex` → zero defects
- [ ] File is self-contained (no `\input`, `\includegraphics`, `\bibliography`)

### Practice Notebook

- [ ] Executes top-to-bottom with `OFFLINE_FALLBACK = True` (no internet)
- [ ] All `assert` statements pass
- [ ] `python -m nbformat.validate dNN_topic.ipynb` exits clean
- [ ] No cell requires > 16 GB VRAM
- [ ] Notebook outputs stripped (nbstripout active — `"outputs": []` in all cells)
- [ ] Terminology grep clean on notebook file

### Traceability

- [ ] Every numeric assert cites the lecture slide where the number is hand-computed
  (`# Ma'ruza [I]-slayd bilan solishtiring`)

### Security

- [ ] No hardcoded tokens, API keys, or credentials in any new or modified file
- [ ] Secrets read via `os.environ.get(...)` or `UserSecretsClient().get_secret(...)`
- [ ] No large data files staged (`corpus/`, `datasets/`, `*.bin`, `*.kv`, etc.)

---

## QA report summary

Paste the one-line gate summary from `course/qa/dNN_report.md`:

> _e.g.: All 5 gates PASS — compile ✓, slide count 25 ✓, archetypes ✓, terminology ✓, notebook offline ✓_

---

## Reviewer checklist

- [ ] QA report shows all gates PASS
- [ ] Artifacts match the approved `course_map.yaml` entry for this day
- [ ] No course content (lecture numbers, assert values) changed without
  a corresponding `course_map.yaml` update
- [ ] No large files accidentally staged
