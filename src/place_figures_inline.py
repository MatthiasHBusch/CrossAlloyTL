"""Move all figure environments from end of manuscript.tex inline
after their first \\ref{...} mention."""
import re
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "manuscript" / "manuscript.tex"

txt = TEX.read_text(encoding="utf-8")

# Extract all figure environments
fig_pat = re.compile(r"\\begin\{figure\}\[!tbp\].*?\\end\{figure\}\s*",
                     re.DOTALL)
figs = fig_pat.findall(txt)


def get_label(fig):
    m = re.search(r"\\label\{fig:(\w+)\}", fig)
    return m.group(1) if m else None


fig_by_label = {get_label(f): f for f in figs}
for label in fig_by_label:
    print(f"Found figure: {label}")

# Remove figure environments from current position
txt_no_figs = fig_pat.sub("", txt)
# Remove banner
txt_no_figs = re.sub(r"%={5,}\s*\n%\s*FIGURES\s*\n%={5,}\s*\n", "",
                     txt_no_figs)


def insert_after_first_ref(text, needle, fig_content):
    """Insert fig_content after the paragraph that contains `needle`."""
    idx = text.find(needle)
    if idx == -1:
        print(f"  (no match for: {needle[:60]})")
        return text
    # Find end of the paragraph: next blank line
    para_end = text.find("\n\n", idx)
    if para_end == -1:
        return text[:idx + len(needle)] + "\n\n" + fig_content + "\n" + text[idx + len(needle):]
    return text[:para_end + 2] + fig_content + "\n" + text[para_end + 2:]


# Order: insert latest refs first, so earlier offsets don't shift
insertions = [
    ("r2mae",        r"\subsection{Cross-model comparison}"),
    ("mgca",         r"Figure~\ref{fig:mgca}"),
    ("per_alloy",    r"\subsection{Cross-model comparison}"),  # place near sec 3.5
    ("alloysim",     r"Figure~\ref{fig:alloysim} shows"),
    ("leakage",      r"Figure~\ref{fig:leakage} shows"),
    ("main",         r"Figure~\ref{fig:main} shows"),
    ("samplecounts", r"Figure~\ref{fig:samplecounts} shows"),
]

# Custom: if same anchor used twice, the second insertion will fall AFTER
# the first figure's closing (since \begin{figure}...\end{figure}\n\n
# creates a new paragraph break). That's fine.

new_txt = txt_no_figs
for label, needle in insertions:
    if label not in fig_by_label:
        print(f"Skipping {label} (not found)")
        continue
    fig_block = fig_by_label[label].rstrip() + "\n"
    new_txt = insert_after_first_ref(new_txt, needle, fig_block)
    print(f"Placed figure: {label}")

TEX.write_text(new_txt, encoding="utf-8")
print("\nDone. Figures inlined.")
