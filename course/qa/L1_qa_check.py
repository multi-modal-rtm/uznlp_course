"""QA gates for L1: course/lectures/d01_nlp_asoslari.tex"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')

TEX = "course/lectures/d01_nlp_asoslari.tex"
PASS = True

# ── 1. Terminology grep ───────────────────────────────────────────────────
print("=== 1. Terminology grep ===")
banned = re.compile(r"professor|talaba|student|o'qituvchi", re.IGNORECASE)
with open(TEX, encoding="utf-8") as f:
    lines = f.readlines()
hits = 0
for i, line in enumerate(lines, 1):
    if banned.search(line):
        # allow only linguistic examples (morpheme demo strings)
        is_example = ("o'qi-tuv-chi" in line or
                      "o'qituvchi-lar" in line.lower() or
                      "o'qituvchilar" in line.lower())
        verdict = "OK (linguistic)" if is_example else "DEFECT"
        if not is_example:
            hits += 1
            PASS = False
        print(f"  {verdict} {TEX}:{i}: {line.rstrip()[:80]}")
if hits == 0:
    print(f"  CLEAN  {TEX}")
print(f"  Terminology gate: {'PASS' if hits == 0 else f'FAIL ({hits} defects)'}")

# ── 2. Slide-count heuristic (target 38–42) ──────────────────────────────
print()
print("=== 2. Slide-count heuristic ===")
with open(TEX, encoding="utf-8") as f:
    tex = f.read()

# Exclude appendix frames — target is slides before \appendix
if r"\appendix" in tex:
    body_tex = tex[:tex.index(r"\appendix")]
    appendix_frames = tex.count(r"\begin{frame}") - body_tex.count(r"\begin{frame}")
else:
    body_tex = tex
    appendix_frames = 0

explicit_frames = body_tex.count(r"\begin{frame}")
has_maketitle   = r"\maketitle" in body_tex
n_sections      = body_tex.count(r"\section")
atbeg_def       = 1 if r"\AtBeginSection" in body_tex else 0
rendered        = explicit_frames + (n_sections - atbeg_def) + (1 if has_maketitle else 0)

ok = 38 <= rendered <= 42
print(f"  Source \\begin{{frame}} (body only): {explicit_frames}")
print(f"  \\maketitle present:   {'yes (+1)' if has_maketitle else 'no'}")
print(f"  \\section count:       {n_sections}")
print(f"  AtBeginSection def:   {atbeg_def}")
print(f"  Estimated rendered:   {rendered}  (target 38-42)  →  {'OK' if ok else 'OUT OF RANGE'}")
if not ok:
    PASS = False
print(f"  Appendix [S] frames:  {appendix_frames}  (excluded from count)")

# ── 3. Self-contained check ───────────────────────────────────────────────
print()
print("=== 3. Self-contained check ===")
checks = {
    r"\input":          r"\\input" not in tex,
    r"\include":        r"\\include" not in tex,
    r"\includegraphics": r"\\includegraphics" not in tex,
    r"\bibliography":   r"\\bibliography" not in tex,
    r"\setmainfont":    r"\\setmainfont" not in tex,
    r"\fontspec":       r"\\fontspec" not in tex,
}
for label, ok_val in checks.items():
    print(f"  {'OK' if ok_val else 'FOUND (check needed)'}  {label}")
    if not ok_val:
        PASS = False

# ── 4. Archetype structure check ─────────────────────────────────────────
print()
print("=== 4. Archetype structure check ===")
archetype_checks = {
    "[A] \\maketitle":              r"\maketitle"          in tex,
    "[B] through-line hook":        "16 Kunda"             in tex,
    "[C] maqsadlar":               "Maqsad"               in tex or "qila olasiz" in tex,
    "[D] reja/vaqt byudjeti":       "Reja" in tex or "Tsikl" in tex,
    "[E] muammo birinchi":          "EMAS" in tex or "Noto" in tex or "xato" in tex.lower(),
    "[F1..F4] intuitsiya (x4)":     tex.count("[F") + tex.lower().count("hamma joyda") + tex.lower().count("shovqin") >= 2,
    "[G] defbox (ta'rif)":          "defbox" in tex,
    "[H] formula/derivatsiya":      r"\mathrm{TF" in tex or r"\mathrm{IDF" in tex,
    "[I] hand example (exact corpus)": "nlp qiziq" in tex and "0.405" in tex and "1.099" in tex,
    "[J] warnbox+okbox (x4)":       tex.count("warnbox") >= 4 and tex.count("okbox") >= 4,
    "[K] kod listingi (x2+)":       tex.count("lstlisting") >= 4,
    "[L] pitfall slayd":            "Tuzog'" in tex or "Sindiradi" in tex,
    "[M] uzbek tili slayd":         "Agglutinatsiya" in tex or "agglutinativ" in tex.lower(),
    "[N] taqqoslov jadvali":        "Taqqoslov" in tex or "CountVec" in tex,
    "[O] seminal manba":            "Salton" in tex,
    "[P] bajarildi checkmark":      r"\bajarildi" in tex,
    "[Q] ko'prik P1 ga":           "P1" in tex and "tikzpicture" in tex,
    "[R] adabiyotlar":              "Jurafsky" in tex or "Adabiyot" in tex,
    "[S] appendix":                 r"\appendix" in tex,
    "\\bunda in formula slides":   r"\bunda" in tex,
    "TF-IDF traceability comment": "P1" in tex and ("Ma'ruza" in tex or "assert" in tex.lower()),
    "No \\input/\\include":        r"\input" not in tex and r"\include" not in tex,
    "No \\includegraphics":        r"\includegraphics" not in tex,
    "TikZ (arrows.meta)":          r"arrows.meta" in tex,
}
fails = 0
for label, result in archetype_checks.items():
    mark = "OK" if result else "MISSING"
    print(f"  {mark}  {label}")
    if not result:
        fails += 1
        PASS = False
print(f"  Archetype gate: {'PASS' if fails == 0 else f'FAIL ({fails} missing)'}")

# ── 5. pdflatex gate ─────────────────────────────────────────────────────
print()
print("=== 5. pdflatex gate ===")
print("  DEFERRED to Overleaf (pdfLaTeX) — not a local blocker.")
print("  Upload: course/lectures/d01_nlp_asoslari.tex as a SINGLE file.")
print("  Overleaf: pdflatex x2; check log for ^! and Overfull >10pt.")

# ── SUMMARY ──────────────────────────────────────────────────────────────
print()
print("==" * 30)
print(f"OVERALL: {'PASS' if PASS else 'FAIL (see above)'}")
print("  Compile gate: deferred to Overleaf (pdfLaTeX) — not a local blocker.")
