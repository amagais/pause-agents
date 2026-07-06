"""Shared manuscript figure styling: ONE Nature-style palette + consistent
bold panel letters across every figure (Figs 2/3/4), so colors are identical
paper-wide rather than re-typed per script.

Palette matches the Nature Medicine reference (Vishwanath et al. 2026): muted
blue / green / purple / orange / red / gray.
"""

from __future__ import annotations

NATURE = {
    "blue":   "#4B7BA2",
    "green":  "#4B8A7B",
    "purple": "#765FA3",
    "orange": "#C87C48",
    "red":    "#B24945",
    "gray":   "#B3B3B3",
}
CYCLE = [NATURE["blue"], NATURE["green"], NATURE["purple"],
         NATURE["orange"], NATURE["red"], NATURE["gray"]]
CONSENSUS = NATURE["gray"]   # consensus / pooled series uses gray

# the 4 generator models (Figs 3/4). gpt-5.4 = blue (reference/deployed model).
GENERATOR_COLORS = {
    "gpt-5.4":  NATURE["blue"],
    "gemma-4":  NATURE["green"],
    "medgemma": NATURE["purple"],
    "qwen-3.6": NATURE["orange"],
}
# the 3 judge families (Fig 2) — kept consistent with the first rendered Fig 2.
JUDGE_COLORS = {
    "deepseek-r1": NATURE["blue"],
    "llama33-70b": NATURE["orange"],
    "o4-mini":     NATURE["red"],
}

_GEN_KEYS = [("gpt", "gpt-5.4"), ("gemma", "gemma-4"),
             ("medgemma", "medgemma"), ("qwen", "qwen-3.6")]
_JUDGE_KEYS = [("deepseek", "deepseek-r1"), ("llama", "llama33-70b"),
               ("o4", "o4-mini")]


def _resolve(label, table, keys, i):
    if label in table:
        return table[label]
    low = str(label).lower()
    # longest-keyword-first so "medgemma" wins over "gemma"
    for kw, canon in sorted(keys, key=lambda k: -len(k[0])):
        if kw in low:
            return table[canon]
    return CYCLE[i % len(CYCLE)]


def generator_color(label, i=0):
    return _resolve(label, GENERATOR_COLORS, _GEN_KEYS, i)


def judge_color(label, i=0):
    return _resolve(label, JUDGE_COLORS, _JUDGE_KEYS, i)


def color_map(labels, kind="generator"):
    """{label: hex} for a list of labels. kind in {'generator','judge'}."""
    fn = generator_color if kind == "generator" else judge_color
    return {lab: fn(lab, i) for i, lab in enumerate(labels)}


def panel_label(ax, letter, x=-0.09, y=1.04, fontsize=15):
    """Bold lowercase panel letter at the top-left corner (Nature style)."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="right")


def scale_fonts(fig, factor=1.3):
    """Uniformly enlarge every text element in a figure — axis labels, tick
    labels, annotations, titles, legend entries, panel letters — by `factor`.

    The plot scripts set font sizes per-call with hard-coded points, so a global
    rcParams bump would not reach them; scaling the realized Text artists does.
    Call right before savefig and pair with bbox_inches='tight' so the enlarged
    text is not clipped at the figure edge. Reviewers flagged the defaults as too
    small for print; factor=1.3 restores legibility without re-laying-out panels.
    """
    import matplotlib.text

    fig.canvas.draw()  # ensure tick labels are realized before scaling
    for t in fig.findobj(matplotlib.text.Text):
        cur = t.get_fontsize()
        if cur:
            t.set_fontsize(cur * factor)
