---
name: lecture-beamer
description: Use whenever creating or editing any lecture slide deck (.tex/.pdf) for the NLP course. Enforces the fixed visual identity, the A–S pedagogical archetype skeleton, and compile-verification gates.
---

# Lecture Deck Production Rules

Every lecture is an instance of the master template at
`assets/00_marquza_shabloni.tex`. Copy it; never rebuild the preamble.

## Immutable visual identity

The preamble blocks marked "asl fayldan o'zgarishsiz" (colors, Boadilla theme
setup, footer, lstset, tcbset, tikz styles) must be copied byte-identical into
every lecture. Never define new colors, never restyle defbox/warnbox/okbox,
never alter the footer. All TikZ diagrams reuse the existing node styles
(rounded rectangles, cnublue/cnubluebg fill, green!12 for the final/goal box).

## Mandatory skeleton

Every lecture contains all archetypes in order (comments [A]–[S] in the
template explain each):
[A] title → [B] recap quiz (2 questions, answers after \pause, zero new
material) → [C] measurable objectives (verbs: keltirib chiqara oladi /
hisoblay oladi / taqqoslay oladi / kodda bog'lay oladi) + prerequisites →
[D] plan with time-budget table → [E] problem-first motivation (show where
the naive solution fails BEFORE naming today's method) → per theory section:
[F] intuition (picture, zero formulas) → [G] formal definition in defbox with
mandatory `\bunda{...}` symbol key → [H] step-by-step derivation (≤ 4 steps
per slide, gray annotation per step, \pause only BETWEEN displayed equations,
never inside align) → [I] hand-worked example on a tiny Uzbek corpus, numbers
computable mentally → [J] "Sizning vazifangiz" (warnbox question → \pause →
okbox answer; same shape as [I], different numbers) → [K] code↔formula bridge
(listing left, mapping table right; comments number the formula steps) →
[L] limitations + one real error with its fix → [M] Uzbek-language
considerations slide (mandatory in every lecture) →
[N] synthesis comparison table (columns = decision criteria) → [O] one seminal
paper + one discussion question assigned for next session → [P] conclusions
that checkmark (\bajarildi) back to [C]'s objectives verbatim → [Q] bridge to
tomorrow's practice: TikZ pipeline of what gets built, which slides ([G],[I])
become code/asserts, "tayyorlab keling" list → [R] references → [S] \appendix
backup slides (math refreshers for [C]'s prerequisites, deeper proofs).

## Content rules

- One idea per slide. Frame titles are full statements ("TF-IDF kam
  uchraydigan so'zni mukofotlaydi"), not labels ("TF-IDF").
- Every formula slide has a `\bunda{}` key defining every symbol.
- A theory block is never more than 3 slides away from an example or task.
- All running examples use Uzbek text; English corpora only when contrasting
  language behavior is the point.
- `\section[Qisqa]{To'liq nom}` — always give short titles so the footer
  navigation never overflows.
- Numbers in [I] and [J] must match the asserts in the paired practice
  notebook exactly. Coordinate via course_map.yaml.

## Verification gate (run before presenting)

```bash
pdflatex -interaction=nonstopmode dNN_topic.tex   # twice
grep -E "^!" dNN_topic.log                         # must be empty
grep -E "Overfull \\\\hbox \([1-9][0-9]" dNN_topic.log  # must be empty (>10pt)
```
Slide count 22–28 before \appendix. Render 3 random pages with
`pdftoppm -png -r 60` and visually inspect for overflow/collisions.
