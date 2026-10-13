"""One chart: the French token tax per tokenizer on the news corpus, against the
character baseline (how much longer French is before any tokenizer touches it)."""

from __future__ import annotations

from pathlib import Path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAR = "#2a78d6"
BASE = "#9a9994"

NEWS = "news (WMT14, human translation)"


def draw(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    news = [r for r in rows if r["corpus"] == NEWS]
    base = next(r for r in news if r["tokenizer"] == "(characters)")["tax"] * 100
    toks = sorted((r for r in news if r["tokenizer"] != "(characters)"),
                  key=lambda r: r["tax"])
    names = [r["tokenizer"] for r in toks]
    vals = [r["tax"] * 100 for r in toks]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(names))
    ax.barh(y, vals, height=0.62, color=BAR, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, f"+{v:.0f}%", va="center", ha="left",
                color=INK, fontsize=17, fontweight="bold", zorder=4)

    ax.axvline(base, color=BASE, lw=2, ls=(0, (4, 3)), zorder=2)
    ax.text(base + 0.6, len(names) - 0.45,
            f"French is {base:.0f}% longer\nin raw characters",
            color=INK_2, fontsize=13, va="bottom", ha="left")

    ax.set_yticks(list(y), names, color=INK, fontsize=16)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_ylim(-0.6, len(names) + 0.4)
    ax.xaxis.set_major_formatter(lambda x, _: f"+{x:.0f}%")
    ax.tick_params(axis="x", colors=INK_2, labelsize=13, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(0.04, 0.955, "The French token tax", color=INK, fontsize=30,
             fontweight="bold", ha="left")
    fig.text(0.04, 0.917,
             "Extra tokens to say the same thing in French instead of English",
             color=INK_2, fontsize=17, ha="left")
    fig.text(0.04, 0.025,
             "3,003 human-translated sentence pairs, WMT14 newstest. "
             "95% bootstrap CIs all within ±1.1 pts.  github.com/AdamosZ/tokenizer-tax",
             color=INK_2, fontsize=11.5, ha="left")

    fig.subplots_adjust(left=0.30, right=0.95, top=0.87, bottom=0.08)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
