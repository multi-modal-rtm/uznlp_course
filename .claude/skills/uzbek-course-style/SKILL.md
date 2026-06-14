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

## Uzbek prose & capitalization style

### Sentence case — binding for all slide/heading text

Frame titles, section headings, tcolorbox block titles, and slide-level
headings use **Uzbek sentence case**: capitalize only the first word of the
title plus proper nouns (O'zbek, NLP, BERT, TF-IDF, BoW, Kaggle, …).
Do NOT capitalize every content word (Title Case). This applies to all
future lectures, notebooks, and briefs.

Concrete examples extracted from L1 (d01_nlp_asoslari.tex, reference file):

| Context | WRONG (machine-generated) | RIGHT (sentence case) |
|---------|---------------------------|----------------------|
| Frame title | `Morfologiya: O'zbek So'z Tuzilishi NLP ni Qiyinlashtiradi` | `Morfologiya: O'zbek so'z tuzilishi NLP ni qiyinlashtiradi` |
| Frame title | `Sintaksis va Semantika: Tartib Muhim, Ma'no Murakkab` | `Sintaksis va semantika: tartib muhim, ma'no murakkab` |
| Frame title | `Sentiment Tahlili: Sharh Musbatmi Yoki Salbiy?` | `Sentiment tahlili: Sharh ijobiymi yoki salbiy?` |
| Section heading | `NLPga Kirish va Tarixi` | `NLPga kirish` |
| tcolorbox title | `O'zbek Hujjat Yordamchisi` | `O'zbek hujjat yordamchisi` |

### Natural Uzbek phrasing — binding for all prose

Avoid mechanical, translation-flavored, or over-formal phrasing. Patterns
observed in L1 edits:

- **Prefer native vocabulary over calques:**
  `ijobiy/salbiy` not `musbat/salbiy`; `kirish/chiqish` not
  `kiritma/chiqarma`; `reja` not `tsikl`; `gap` not `hujjat` when
  referring to a corpus sentence.
- **Inclusive first-person plural in frame-level text:**
  `"o'rganamiz"`, `"ko'ramiz"`, `"hisoblaymiz"` — not commanding second
  person (`"o'rganasiz"`, `"ko'ring"`). This matches the peer-to-peer
  tone rule.
- **Complete thoughts, not bare labels:**
  Frame titles should be statements or questions that could stand alone.
  `"Xom matn shovqinga to'la bo'ladi — tozalash shart"` reads better
  than the label `"Xom Matn Shovqin To'la — Tozalash Shart"`.
- **Vary sentence structure:** avoid the pattern `Verb: long noun clause`
  for every objective bullet — integrate the verb into a full sentence.

### Verification

This is a **prose-quality goal verified by human review on Overleaf**,
not by an automated grep gate. After compiling a new deck, the author
reads frame titles top-to-bottom and checks: (a) no word is capitalized
purely because it appears mid-title, and (b) phrasing sounds spoken, not
machine-translated.

## Assert and error message style

Success: short, specific, affirming — `"To'g'ri! TF-IDF matritsa shakli
(3, 7) — lug'at 7 so'zdan iborat."` Failure: state what to re-check and
where — `"Shakl mos kelmadi: lug'at hajmini tekshiring (Ma'ruza,
[I]-slayd)."` Never exclamation-heavy, never blame.
