---
name: uzbek-course-style
description: Use whenever writing any tinglovchi-facing text — slides, notebooks, briefs, comments, asserts, READMEs — for the NLP course. Enforces audience terminology, Uzbek Latin-script conventions, and field-neutral wording.
---

# Uzbek Course Style & Terminology

## Audience terminology (zero tolerance)

- Correct: **tinglovchi / tinglovchilar** (Uzbek), **participant(s)**
  (English). Possessive framing is encouraged: "sizning loyihangiz",
  "o'z korpusingiz".
- FORBIDDEN in any task, project text, slide, notebook, comment, or assert
  message: `professor`, `professor-o'qituvchi`, `talaba`, `student`,
  `o'qituvchi` (as audience reference; "o'qituvchi" may appear only inside
  linguistic example words like "o'qituvchilarimizdan").
- Examples and projects must be field-neutral: prefer corpora any
  professional relates to (news, official documents, the tinglovchi's OWN
  chosen documents) over academia-specific framing.
- QA grep (run on every artifact):
  `grep -rinE "professor|talaba|student|o'qituvchi" <file>`
  — every hit must be a linguistic example (`o'qituvchi` only) or it is a
  defect.

## Uzbek Latin script conventions

- The apostrophe in o', g' is U+2019 (') or ASCII (') — pick ASCII (')
  consistently in code/corpora so tokenizer regexes behave; in LaTeX prose
  either works, do not mix within one file.
- Remember NLP consequences: standard tokenizers split "so'z" wrongly —
  this is a feature of the course (taught Day 2), so example code must use
  the course's Uzbek-aware regex, not naive word_tokenize, after Day 2.
- Established terms: give Uzbek first, English in parentheses on first use
  per document: "so'z qopi (Bag of Words)", "ichki vektorlar (embeddings)",
  "diqqat mexanizmi (attention)". Afterwards the Uzbek or the abbreviation.
- Numbers/dates follow Uzbek convention in prose ("15-iyun", "2-hafta").
- Tone: professional, warm, peer-to-peer ("hisoblaymiz", "ko'ramiz" — first
  person plural), never condescending; tinglovchilar are experienced
  professionals new to THIS field only.

## Assert and error message style

Success: short, specific, affirming — `"To'g'ri! TF-IDF matritsa shakli
(3, 7) — lug'at 7 so'zdan iborat."` Failure: state what to re-check and
where — `"Shakl mos kelmadi: lug'at hajmini tekshiring (Ma'ruza,
[I]-slayd)."` Never exclamation-heavy, never blame.
