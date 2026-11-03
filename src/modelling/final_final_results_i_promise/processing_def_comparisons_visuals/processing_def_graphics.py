"""Plots of the test-time defence chains against no defence, from src/modelling/final_final_results_i_promise/.

Reads two files from that folder (made by src/modelling/test_all.py):
  <metric>.txt   one row per pre-processing chain x model x post-processing chain, with one column
                 per test set (clean_test, pgd_test, pixle_test, pgdpixle_test). Default: f1_macro.txt.
  runtime.txt    time per image (ms) of the whole pipeline for each row, mean over the 4 test sets.

For one model it draws three charts. Every panel has a gray line at no defence (pre = none, post = none),
so each chain is always compared with the undefended model.

  pre          the pre-processing chains alone (post = none)
  post         the 15 post-processing chains alone (pre = none), grouped by defence set
  combination  each pre-processing chain + each of the 15 post-processing chains, as one defence each

Panels: the metric on the 4 test sets, then "Mean (attacked)" = the mean of the metric over the PGD, Pixle
and PGD + Pixle sets (blue = better than no defence, red = worse, number = change), then time per image.

ConvAE is not a pre-processing defence, so pre chains that contain "convae" are left out (SKIP_PRE).

In a notebook in this folder:

  import processing_def_graphics as G
  G.plot_all("baseline")                       # shows the 3 charts inline; nothing is saved
  G.plot_all("grl", metric="accuracy")         # the same charts with another metric file

From a shell (from this folder):

  python processing_def_graphics.py --tag baseline             # opens the charts
  python processing_def_graphics.py --tag all --out .          # saves the PNGs here

processing_def_plots.ipynb (same folder) runs plot_all for every model and saves the PNGs in this folder.
"""
import argparse
import re
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1]  # src/modelling/final_final_results_i_promise
TAGS = {  # model name in the result files -> model file
    "baseline": "best_baseline.keras",
    "robust": "best_robust.keras",
    "grl": "grl.keras",
    "adversarial_retrained": "best_adversarial_retrained.keras",
    "convae_rn50": "best_convae_rn50.keras",
}
ALIASES = {"adv_retrained": "adversarial_retrained"}
NONE = "none"
ATTACKED = ["pgd_test", "pixle_test", "pgdpixle_test"]
MEAN_ATTACKED = "mean_attacked"  # mean of the metric over the ATTACKED test sets
SETS = [("clean_test", "Clean"), ("pgd_test", "PGD"), ("pixle_test", "Pixle"), ("pgdpixle_test", "PGD + Pixle"),
        (MEAN_ATTACKED, "Mean (attacked)")]
POST_NAMES = ["noise", "highconf", "revsig"]
SKIP_PRE = ("convae",)  # ConvAE is not a pre-processing defence
METRIC = "f1_macro"
LOWER_IS_BETTER = {"cross_entropy_loss"}
EPS = 1e-9  # values are stored to 6 decimals; any stored difference counts as a change

# Colours (validated reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, RED = "#2a78d6", "#e34948"          # better / worse than no defence
TIME_BAR, TIME_DOT = "#b5b4ae", INK2      # neutral: time is a cost on its own scale
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
}


def chains(names):
    """Each single defence, then every ordered pair and triple (no 'none')."""
    out = []
    for k in range(1, len(names) + 1):
        out += [" -> ".join(c) for c in permutations(names, k)]
    return out


def label(c):
    return "no defence" if c == NONE else c.replace(" -> ", " → ")


def metric_label(metric):
    return {"f1_macro": "Macro F1", "accuracy": "Accuracy"}.get(metric, metric.replace("_", " ").capitalize())


@lru_cache(maxsize=None)
def read_table(path):
    """A fixed-width results table -> {(pre, model, post): {column: value}}."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    dash = next(i for i, ln in enumerate(lines) if ln.startswith("---"))
    head = re.split(r"\s{2,}", lines[dash - 1].strip())
    rows = {}
    for ln in lines[dash + 1:]:
        if not ln.strip():
            continue
        cells = re.split(r"\s{2,}", ln.strip())
        r = dict(zip(head, cells))
        rows[(r["Preprocessing"], r["Model"], r["Postprocessing"])] = r
    return rows


class Results:
    """The rows of one model, with the values the charts need."""

    def __init__(self, tag, results_dir=RESULTS, metric=METRIC):
        tag = ALIASES.get(tag, tag)
        path = Path(results_dir) / f"{metric}.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        table = read_table(str(path))
        times = read_table(str(Path(results_dir) / "runtime.txt"))
        self.rows = {(p, q): r for (p, m, q), r in table.items() if m == tag}
        if not self.rows:
            models = sorted({m for _, m, _ in table})
            raise KeyError(f"No rows for model '{tag}' in {path.name}. Models in the file: {', '.join(models)}")
        self.times = {(p, q): r for (p, m, q), r in times.items() if m == tag}
        self.model = TAGS.get(tag, tag)
        self.metric = metric
        self.sign = -1 if metric in LOWER_IS_BETTER else 1
        self.pre = [p for p in dict.fromkeys(p for p, _ in self.rows)
                    if p != NONE and not any(s in p for s in SKIP_PRE)]
        self.post = [q for q in chains(POST_NAMES) if (NONE, q) in self.rows]

    def value(self, pre, post, s):
        if s == MEAN_ATTACKED:
            return sum(self.value(pre, post, a) for a in ATTACKED) / len(ATTACKED)
        return float(self.rows[(pre, post)][s])

    def ms(self, pre, post):
        """Whole pipeline (pre chain + model + post chain), ms per image."""
        return float(self.times[(pre, post)]["Avg Time per Inference (ms)"])


def post_layout(posts, y=0.0):
    """Post chains grouped by defence set: single defences, each pair set, then all three.
    Returns rows [(y, chain)], headers [(y, text)] and the next free y."""
    groups = [("single defences", [c for c in posts if " -> " not in c])]
    for k in (2, 3):
        sets_ = []
        for c in posts:
            parts = c.split(" -> ")
            if len(parts) == k and set(parts) not in sets_:
                sets_.append(set(parts))
        for s in sets_:
            members = [c for c in posts if len(c.split(" -> ")) == k and set(c.split(" -> ")) == s]
            title = "{" + ", ".join(n for n in POST_NAMES if n in s) + "}" + (" (all orders)" if k == 3 else "")
            groups.append((title, members))
    rows, headers = [], []
    for title, members in groups:
        headers.append((y, title))
        y += 1
        for c in members:
            rows.append((y, c))
            y += 1
        y += 0.4
    return rows, headers, y


def _value_panel(ax, R, title, base, values, rings=None, lim=(0.0, 1.0)):
    """values: [(y, v)]. Gray line at no defence, bar from it to each value, signed change label.
    rings: optional [(y, v)] drawn as hollow markers (the pre chain alone)."""
    ax.axvline(base, color=INK2, lw=1.2, zorder=1)
    ring_at = dict(rings or [])
    span = lim[1] - lim[0]
    for y, v in values:
        d = v - base
        col = MUTED if abs(d) <= EPS else (BLUE if d * R.sign > 0 else RED)
        ax.plot([base, v], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        if y in ring_at:
            ax.scatter([ring_at[y]], [y], s=40, facecolor="none", ec=INK2, lw=1, zorder=4)
        ax.scatter([v], [y], s=34, color=col, ec=SURFACE, lw=1.5, zorder=3)
        if abs(d) > EPS:
            # next to the dot; if the ring is in the way (close, on the same side), outside both markers
            r = ring_at.get(y, v)
            ring_in_way = (r > v if d > 0 else r < v) and abs(r - v) < 0.15 * span
            edge = (max(v, r) if d > 0 else min(v, r)) if ring_in_way else v
            x = edge + 0.02 * span if d > 0 else edge - 0.02 * span
            ax.text(x, y, f"{d:+.3f}" if abs(d) >= 0.0005 else f"{d:+.4f}", va="center", ha="left" if d > 0 else "right", fontsize=6.5, color=INK2)
    ax.set_xlim(lim[0] - 0.06 * span, lim[1] + 0.14 * span)
    ax.set_title(f"{title}: no defence {base:.3f}", loc="left", fontsize=9,
                 fontweight="bold" if title.startswith("Mean") else "normal")
    ax.set_xlabel(f"Mean {metric_label(R.metric)} (3 attacked sets)" if title.startswith("Mean")
                  else metric_label(R.metric))


def _time_panel(ax, base, values, log):
    """values: [(y, ms)]. Log scale when a chain is many times slower than no defence."""
    ax.axvline(base, color=INK2, lw=1.2, zorder=1)
    top = max(v for _, v in values)
    if log:
        ax.set_xscale("log")
        ax.set_xlim(min(base, min(v for _, v in values)) * 0.8, max(base, top) * 5)
        ticks = [t for t in (2, 5, 10, 20, 50, 100, 200, 500, 1000) if ax.get_xlim()[0] <= t <= top * 1.5]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_xlim(0, max(base, top) * 1.45)
    for y, v in values:
        ax.plot([base, v], [y, y], color=TIME_BAR, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([v], [y], s=34, color=TIME_DOT, ec=SURFACE, lw=1.5, zorder=3)
        if log:
            r = v / base
            txt = f"{v:.1f} ms (×{r:.0f})" if r >= 1.95 else f"{v:.1f} ms (×{r:.1f})"
            x, ha = (v * 1.12, "left") if v >= base else (v / 1.12, "right")
        else:
            txt = f"{v:.1f} ms ({v - base:+.1f})"
            x, ha = (v + top * 0.03, "left") if v >= base else (v - top * 0.03, "right")
        ax.text(x, y, txt, va="center", ha=ha, fontsize=7, color=INK2)
    ax.set_title(f"Time: no defence {base:.1f} ms", loc="left", fontsize=9)


def _figure(n_rows_height):
    fig, axes = plt.subplots(1, len(SETS) + 1, figsize=(22.5, max(0.28 * n_rows_height + 1.9, 3.2)), sharey=True,
                             gridspec_kw={"wspace": 0.12, "width_ratios": [1] * len(SETS) + [1.2]})
    return fig, axes


def _limits(R):
    vals = [R.value(p, q, s) for (p, q) in R.rows if p == NONE or p in R.pre for s, _ in SETS]
    return min(0.0, min(vals)), max(1.0, max(vals))


def _finish(axes, ticks, tick_labels, bold, italic, height, separators):
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for y in separators:
            ax.axhline(y, color=GRID, lw=0.8)
    axes[0].set_yticks(ticks, tick_labels, fontsize=8)
    for t, y in zip(axes[0].get_yticklabels(), ticks):
        if y in italic:
            t.set_fontstyle("italic")
            t.set_color(INK)
        elif y in bold:
            t.set_fontweight("bold")
            t.set_color(INK)
    axes[0].set_ylim(height - 0.6, -0.6)


TIME_NOTE = ("Time: ms per image for the whole pipeline (mean over the 4 test sets, 3 runs). In each group the row "
             "with post = none ran first, so it can include some start-up time.")


def _title(fig, R, what, extra=""):
    better = "lower" if R.sign < 0 else "higher"
    fig.suptitle(f"{R.model}: {what} vs no defence (gray line in each panel). {metric_label(R.metric)}: blue = {better} "
                 f"(better), red = worse, number = change.{extra}\n{TIME_NOTE}", x=0.01, ha="left", fontsize=10, y=1.0)


def plot_pre(tag, results_dir=RESULTS, metric=METRIC):
    """The pre-processing chains alone (post = none) vs no defence."""
    R = Results(tag, results_dir, metric)
    rows = [(float(i), c) for i, c in enumerate(R.pre)]
    lim = _limits(R)
    with plt.rc_context(STYLE):
        fig, axes = _figure(len(rows))
        for ax, (s, title) in zip(axes, SETS):
            _value_panel(ax, R, title, R.value(NONE, NONE, s), [(y, R.value(c, NONE, s)) for y, c in rows], lim=lim)
        base = R.ms(NONE, NONE)
        _time_panel(axes[-1], base, [(y, R.ms(c, NONE)) for y, c in rows], log=True)
        axes[-1].set_xlabel("ms per image (log scale): pre chain + model")
        _finish(axes, [y for y, _ in rows], [label(c) for _, c in rows], set(), set(), len(rows), [])
        _title(fig, R, "pre-processing chains alone")
        fig.subplots_adjust(top=0.72 if len(rows) < 4 else 0.86)
    return fig


def plot_post(tag, results_dir=RESULTS, metric=METRIC):
    """The 15 post-processing chains alone (pre = none) vs no defence, grouped by defence set."""
    R = Results(tag, results_dir, metric)
    rows, headers, height = post_layout(R.post)
    lim = _limits(R)
    with plt.rc_context(STYLE):
        fig, axes = _figure(height)
        for ax, (s, title) in zip(axes, SETS):
            _value_panel(ax, R, title, R.value(NONE, NONE, s), [(y, R.value(NONE, c, s)) for y, c in rows], lim=lim)
        base = R.ms(NONE, NONE)
        _time_panel(axes[-1], base, [(y, R.ms(NONE, c)) for y, c in rows], log=False)
        axes[-1].set_xlabel("ms per image (linear): model + post chain")
        ticks = [y for y, _ in headers] + [y for y, _ in rows]
        tick_labels = [t for _, t in headers] + [f"    {label(c)}" for _, c in rows]
        _finish(axes, ticks, tick_labels, set(), {y for y, _ in headers}, height, [y - 0.7 for y, _ in headers[1:]])
        _title(fig, R, "post-processing chains alone",
               " Rows are grouped by defence set, so the same defences in a different order are next to each other.")
        fig.subplots_adjust(top=0.9)
    return fig


def plot_combined(tag, results_dir=RESULTS, metric=METRIC):
    """Each pre chain + each of the 15 post chains as one defence, grouped by pre chain.
    A hollow ring marks the pre chain alone, so the effect of the post chain is also visible."""
    R = Results(tag, results_dir, metric)
    rows, ticks, tick_labels, heads, subheads, seps, y = [], [], [], [], set(), [], 0.0
    for a in R.pre:
        if heads:
            seps.append(y - 0.75)
        heads.append(y)
        ticks.append(y)
        tick_labels.append(f"{label(a)} +")
        prow, phead, y = post_layout(R.post, y + 1)
        rows += [(yy, a, b) for yy, b in prow]
        for yy, t in phead:
            ticks.append(yy)
            tick_labels.append(f"  {t}")
            subheads.add(yy)
        for yy, b in prow:
            ticks.append(yy)
            tick_labels.append(f"      {label(b)}")
        y += 0.3
    lim = _limits(R)
    with plt.rc_context(STYLE):
        fig, axes = _figure(y)
        for ax, (s, title) in zip(axes, SETS):
            _value_panel(ax, R, title, R.value(NONE, NONE, s), [(yy, R.value(a, b, s)) for yy, a, b in rows],
                         rings=[(yy, R.value(a, NONE, s)) for yy, a, _ in rows], lim=lim)
        base = R.ms(NONE, NONE)
        _time_panel(axes[-1], base, [(yy, R.ms(a, b)) for yy, a, b in rows], log=True)
        axes[-1].set_xlabel("ms per image (log scale): pre chain + model + post chain")
        _finish(axes, ticks, tick_labels, set(heads), subheads, y, seps)
        legend = [plt.Line2D([], [], color=INK2, lw=1.2, label="no defence"),
                  plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, label="pre chain alone (post = none)"),
                  plt.Line2D([], [], marker="o", ls="", color=BLUE, label="combination: better than no defence"),
                  plt.Line2D([], [], marker="o", ls="", color=RED, label="combination: worse"),
                  plt.Line2D([], [], marker="o", ls="", color=MUTED, label="combination: same as no defence")]
        fig.subplots_adjust(top=0.88)
        fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0.9), ncol=5, frameon=False, fontsize=8)
        _title(fig, R, "each pre chain + each post chain as one defence")
    return fig


def plot_all(tag, results_dir=RESULTS, out=None, metric=METRIC):
    """The pre, post and combination charts for one model, as {"pre", "post", "combination"} figures.
    With out=<folder>, also saves them as <model>_pre.png, <model>_post.png and <model>_combination.png,
    where <model> is the model file name without .keras (for example best_baseline_pre.png)."""
    figs = {"pre": plot_pre(tag, results_dir, metric), "post": plot_post(tag, results_dir, metric),
            "combination": plot_combined(tag, results_dir, metric)}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(TAGS.get(ALIASES.get(tag, tag), tag)).stem
        for name, fig in figs.items():
            fig.savefig(out / f"{stem}_{name}.png", dpi=160, bbox_inches="tight")
    return figs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="baseline", help=f"one of {', '.join(TAGS)}, or 'all'")
    ap.add_argument("--results", default=str(RESULTS), help="folder with <metric>.txt and runtime.txt")
    ap.add_argument("--metric", default=METRIC, help="metric file name without .txt (default f1_macro)")
    ap.add_argument("--out", help="save PNGs to this folder instead of opening them")
    args = ap.parse_args()
    tags = list(TAGS) if args.tag == "all" else [args.tag]
    for tag in tags:
        plot_all(tag, args.results, args.out, args.metric)
        if args.out:
            plt.close("all")
            print(f"saved {tag} charts to {args.out}")
    if not args.out:
        plt.show()


if __name__ == "__main__":
    main()
