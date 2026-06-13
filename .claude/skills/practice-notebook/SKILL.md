---
name: practice-notebook
description: Use whenever creating or editing any practice session notebook (.ipynb) for the NLP course. Enforces the 80-minute end-to-end project architecture, periphery/core split, PRIMM, faded scaffolding, asserts, and checkpoints.
---

# Practice Notebook Production Rules

Every practice session builds one complete end-to-end mini-project in 80
minutes AND contributes one module to the tinglovchi's cumulative capstone.
The teaching method, fixed for all 16 sessions:
**run the whole → interrogate the given → fade the core → integrate into
theirs → close with a question.**

## Periphery/core split (decide first, record in course_map.yaml)

For each session, classify every part of the day's pipeline:
- **Periphery**: needed for end-to-end completeness but not today's learning
  goal (e.g., Day 2: web scraping, PDF extraction). Delivered as complete,
  working, commented code. Tinglovchilar never write it — they predict,
  run, investigate, and modify it.
- **Core**: the day's concept (e.g., Day 2: BoW, TF-IDF). Delivered with
  fading scaffolds until produced independently.

## Mandatory notebook structure (sections in this order)

0. **Sarlavha** (markdown): topic; link to yesterday's lecture PDF; today's
   objectives copied from lecture archetype [C]; which capstone module gets
   built; estimated timing per section.
1. **Muhit tekshiruvi** (code): pinned `pip install` versions; GPU assert
   only on GPU days (`assert torch.cuda.is_available()`); define
   `OFFLINE_FALLBACK = True` flag; print versions.
2. **Yaxlit natija** (~8 min): the FINISHED pipeline runs top-to-bottom in
   one or two cells and prints the final artifact. No explanation yet —
   destination first.
3. **Tayyor kod bloki — PRIMM** (~17 min): periphery code with, per chunk:
   a *Bashorat qiling* markdown prompt (predict the output before running),
   the runnable cell, 2–3 *Tekshiring* investigation questions, and one
   *O'zgartiring* modification task that personalizes it (point the scraper
   at the tinglovchi's own chosen page/PDF). Personalization is mandatory —
   their corpus, their capstone.
4. **Asosiy mavzu — so'nuvchi tayanch** (~35 min), three stages:
   - *Namuna* (I do): instructor-style worked cell reproducing the lecture's
     hand example, ending with an assert against the hand-computed number,
     commented `# Ma'ruza [I]-slayd bilan solishtiring`.
   - *Birgalikda* (we do): completion cells — only the conceptually critical
     lines blanked with `# === SIZNING KODINGIZ (taxminan N qator) ===`;
     immediately followed by a self-check assert cell with an encouraging
     Uzbek success message.
   - *Mustaqil* (you do): apply to their own corpus from section 3, no
     scaffold; assert checks structural properties (shapes, ranges), not
     exact values.
5. **Loyihaga ulash** (~13 min): refactor the day's working code into the
   capstone module file (function signatures from capstone/contracts), save,
   and git commit+push. Provide the exact commands.
6. **Tadqiqot savoli + yakun** (~7 min): one open mini-experiment with a
   genuinely non-obvious answer, ideally Uzbek-specific; then a one-line
   markdown exit ticket: "Bugun eng tushunarsiz qolgan narsa nima?"

## Status-protection conventions (this audience will not struggle publicly)

- ALL correctness feedback via asserts — never "raise your hand if".
- After every section boundary, a **checkpoint cell** loads a saved
  intermediate artifact (`if OFFLINE_FALLBACK or rerun: corpus = load(...)`)
  so anyone behind rejoins instantly with working state.
- Error messages in asserts must say what to re-check, kindly, in Uzbek.

## Kaggle execution rules

See skill `kaggle-hardware` for ceilings. Additionally:
- Every notebook must run end-to-end with `OFFLINE_FALLBACK = True`, using
  bundled checkpoint data in `dNN_checkpoints/` — internet-dependent cells
  (scraping, downloads) wrap in try/except that falls back to the bundle.
- No cell may run > 5 minutes on the free tier; long training uses reduced
  epochs with a comment giving the full-scale setting.

## Verification gate

`jupyter nbconvert --to notebook --execute` with OFFLINE_FALLBACK=True must
complete with zero errors; all asserts pass; `nbformat` validation clean;
forbidden-term grep clean; every blanked region has a paired assert.
Solutions notebook: produce `dNN_topic_SOLUTIONS.ipynb` (all blanks filled)
alongside each practice notebook; it must also execute cleanly.
