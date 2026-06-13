"""QA script for Day 1 orientation materials."""
import sys, re, ast
sys.stdout.reconfigure(encoding='utf-8')

PASS = True

print("=== 1. nbformat validate ===")
import nbformat
with open("day1_orientation/d01_orientatsiya.ipynb", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)
print(f"  parse: OK  |  cells: {len(nb.cells)}")
src_all = "".join(c.source for c in nb.cells if c.cell_type == "code")
has_fallback = "OFFLINE_FALLBACK" in src_all
print(f"  OFFLINE_FALLBACK: {'OK' if has_fallback else 'MISSING'}")
if not has_fallback:
    PASS = False

print()
print("=== 2. Terminology grep ===")
files = [
    "capstone/SPEC.md",
    "capstone/contracts.py",
    "day1_orientation/d01_orientatsiya.ipynb",
    "day1_orientation/d01_kirish.tex",
    "day1_orientation/HISOB_YARATISH.md",
    "assessments/gen_pre_course.py",
]
banned = re.compile(r"professor|talaba|student|o'qituvchi", re.IGNORECASE)
total_hits = 0
for fpath in files:
    with open(fpath, encoding="utf-8") as f:
        text = f.read()
    hits = []
    for i, line in enumerate(text.splitlines()):
        if banned.search(line):
            is_example = "o'qi-tuv-chi" in line or "oqituvchi-lar" in line.lower()
            hits.append((i + 1, line.strip(), is_example))
    if hits:
        for lineno, line, ok in hits:
            verdict = "OK (linguistic example)" if ok else "DEFECT"
            if not ok:
                total_hits += 1
                PASS = False
            print(f"  {verdict} {fpath}:{lineno}: {line[:70]}")
    else:
        print(f"  CLEAN  {fpath}")
print(f"  Terminology gate: {'PASS' if total_hits == 0 else f'FAIL ({total_hits} defects)'}")

print()
print("=== 3. contracts.py AST parse ===")
with open("capstone/contracts.py", encoding="utf-8") as f:
    csrc = f.read()
try:
    ast.parse(csrc)
    classes = [n.name for n in ast.walk(ast.parse(csrc)) if isinstance(n, ast.ClassDef)]
    funcs = [n.name for n in ast.walk(ast.parse(csrc)) if isinstance(n, ast.FunctionDef)]
    print(f"  AST: OK  |  classes: {len(classes)}  |  functions: {len(funcs)}")
    print(f"  Classes: {classes}")
except SyntaxError as e:
    print(f"  SYNTAX ERROR: {e}")
    PASS = False

print()
print("=== 4. d01_kirish.tex slide count ===")
with open("day1_orientation/d01_kirish.tex", encoding="utf-8") as f:
    tex = f.read()
# Rendered slide count = maketitle (1) + AtBeginSection frames (1 per \section)
#                      + explicit \begin{frame} in body (excluding AtBeginSection def)
explicit_frames = tex.count(r"\begin{frame}")
has_maketitle = r"\maketitle" in tex
n_sections = tex.count(r"\section")
# AtBeginSection is defined once (\begin{frame} inside it counted above already)
# but it fires once per \section in output → subtract 1 definition, add n_sections
# The AtBeginSection definition contributes 1 to explicit_frames count
atbeginsection_def = 1 if r"\AtBeginSection" in tex else 0
rendered = explicit_frames + (n_sections - atbeginsection_def) + (1 if has_maketitle else 0)
ok = 12 <= rendered <= 16
print(f"  Source \\begin{{frame}}: {explicit_frames}")
print(f"  \\maketitle: {'+1' if has_maketitle else '0'}")
print(f"  \\section (AtBeginSection fires): {n_sections}")
print(f"  Estimated rendered slides: {rendered}  (target 12-16)  ->  {'OK' if ok else 'OUTSIDE RANGE'}")
if not ok:
    PASS = False

print()
print("=== 5. pdflatex gate ===")
print("  DEFERRED to Overleaf (pdfLaTeX) — not a local blocker.")
print("  Upload d01_kirish.tex as a single file; no \\input/\\include/\\includegraphics deps.")
print("  Overleaf compile: pdflatex twice, check log for ^! and Overfull >10pt.")

print()
print("=== 6. Archetype structure check (d01_kirish.tex) ===")
# Orientation deck uses [A][C][D][Q][R] only.
# [A] = \maketitle; [C][D][Q][R] = specific frame titles or \section content.
archetype_checks = {
    "[A] maketitle":         r"\maketitle"               in tex,
    "[C] maqsadlar frame":   "Maqsad"                   in tex,
    "[D] reja/tuzilma":      "Ishlaydi"                  in tex or "Reja" in tex,
    "[Q] kapstone pipeline": "Kapstone"                  in tex and "tikzpicture" in tex,
    "[R] manbalar":          "Manbalar"                  in tex or "Adabiyot" in tex,
    "TikZ pipeline present": "tikzpicture"               in tex,
    "Beamer tcolorbox":      "tcolorbox"                 in tex,
    "No \\input deps":       r"\input"                  not in tex,
    "No \\includegraphics":  r"\includegraphics"        not in tex,
}
for label, ok in archetype_checks.items():
    mark = "OK" if ok else "MISSING"
    print(f"  {mark}  {label}")
    if not ok:
        PASS = False

print()
print("=== 7. Assessment files exist ===")
import os
for f in ["assessments/pre_course.docx", "assessments/pre_course.xlsx"]:
    exists = os.path.isfile(f)
    size = os.path.getsize(f) if exists else 0
    print(f"  {'OK' if exists else 'MISSING'}  {f}  ({size} bytes)")
    if not exists:
        PASS = False

print()
print("==" * 25)
print(f"OVERALL: {'PASS' if PASS else 'FAIL (see above)'}")
print("  Compile gate: deferred to Overleaf (pdfLaTeX) — not a local blocker.")
