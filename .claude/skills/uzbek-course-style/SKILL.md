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
title plus proper nouns and established technical terms (see term list
below). Do NOT capitalize every content word (Title Case). Applies to all
lectures, notebooks, and briefs from L2 onward. Reference file:
`course/lectures/d01_nlp_asoslari.tex`.

#### Proper-noun / term boundary — tiebreaker

Established model, method, and library names keep their **conventional
capitalization** wherever they appear in a title or sentence, even mid-title.
Uzbek words around them follow sentence case.

Canonical list (not exhaustive — use the published spelling):
`BERT`, `TF-IDF`, `Word2Vec`, `FastText`, `GloVe`, `Naive Bayes`,
`Transformer` (when used as the architecture name), `NLP`, `RNN`, `LSTM`,
`BoW`, `RAG`, `CNN`, `GPT`, `spaCy`, `Kaggle`.

Examples:

| Context | WRONG | RIGHT |
|---------|-------|-------|
| Frame title | `Naive bayes klassifikatori bilan ishlash` | `Naive Bayes klassifikatori bilan ishlash` |
| Frame title | `Bert fine-tuning: O'zbek matni uchun sozlash` | `BERT fine-tuning: O'zbek matni uchun sozlash` |

#### Colon rule

After a colon in a frame title or heading, **capitalize the first word of
the new phrase** (the post-colon part is treated as a sub-title or
independent clause). Uzbek sentence-case still applies within that phrase.

Examples from L1 (both patterns correct):

| Context | RIGHT |
|---------|-------|
| Frame title | `Morfologiya: O'zbek so'z tuzilishi NLP ni qiyinlashtiradi` |
| Frame title | `Sentiment tahlili: Sharh ijobiymi yoki salbiy?` |
| Frame title | `Sintaksis va semantika: Tartib muhim, ma'no murakkab` |
| Frame title | `Sizning vazifangiz: Matnni qo'lda tozalang` |

Full L1 examples table (sentence case, L1 reference file):

| Context | WRONG (machine-generated) | RIGHT (sentence case) |
|---------|---------------------------|----------------------|
| Frame title | `Morfologiya: O'zbek So'z Tuzilishi NLP ni Qiyinlashtiradi` | `Morfologiya: O'zbek so'z tuzilishi NLP ni qiyinlashtiradi` |
| Frame title | `Sintaksis va Semantika: Tartib Muhim, Ma'no Murakkab` | `Sintaksis va semantika: Tartib muhim, ma'no murakkab` |
| Frame title | `Sentiment Tahlili: Sharh Musbatmi Yoki Salbiy?` | `Sentiment tahlili: Sharh ijobiymi yoki salbiy?` |
| Section heading | `NLPga Kirish va Tarixi` | `NLPga kirish` |
| tcolorbox title | `O'zbek Hujjat Yordamchisi` | `O'zbek hujjat yordamchisi` |

### Locked sentiment labels — project-wide

**`ijobiy`** = positive sentiment. **`salbiy`** = negative sentiment.
These are NOT interchangeable with `musbat`/`manfiy`.

These labels are locked to L2's [I2] hand-worked example (Naive Bayes
classification result: test document classified `ijobiy`) and to P2's
first assert:
```python
assert result == "ijobiy"   # Ma'ruza L2 [I2]-slayd bilan solishtiring
```
Every future slide, notebook cell, assert message, and inline comment that
refers to sentiment polarity **must** use `ijobiy`/`salbiy`. Using `musbat`
or `manfiy` breaks label consistency and will cause P2's assert to fail.

### Natural Uzbek phrasing — binding for all prose

Avoid mechanical, translation-flavored, or over-formal phrasing. Patterns
established in L1 edits:

- **Prefer native vocabulary over calques:**
  `kirish/chiqish` not `kiritma/chiqarma`; `reja` not `tsikl`; `gap`
  not `hujjat` when referring to a corpus sentence.
- **Default narrative voice — first-person plural:**
  `"o'rganamiz"`, `"ko'ramiz"`, `"hisoblaymiz"`, `"quramiz"` throughout
  all slide prose and notebook markdown. This is the course's canonical
  voice, not a one-off edit. It reinforces the peer-to-peer tone rule
  already in this skill. Second-person command forms (`"o'rganasiz"`,
  `"ko'ring"`) are reserved for explicit audience-task callouts only
  (e.g., the `[J]` "Sizning vazifangiz" frame).
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
purely because it appears mid-title, (b) post-colon word is capitalized,
(c) technical terms use their canonical spelling, and (d) phrasing sounds
spoken, not machine-translated.

## Assert and error message style

Success: short, specific, affirming — `"To'g'ri! TF-IDF matritsa shakli
(3, 7) — lug'at 7 so'zdan iborat."` Failure: state what to re-check and
where — `"Shakl mos kelmadi: lug'at hajmini tekshiring (Ma'ruza,
[I]-slayd)."` Never exclamation-heavy, never blame.
