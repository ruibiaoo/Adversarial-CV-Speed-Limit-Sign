"""Plots of the test-time defence chains against no defence, from src/modelling/final_final_results_i_promise/.

Reads two files from that folder (made by src/modelling/test_all.py):
  <metric>.txt   one row per pre-processing chain x model x post-processing chain, with one column
                 per test set (clean_test, pgd_test, pixle_test, pgdpixle_test). Default: f1_macro.txt.
  runtime.txt    Avg Time per Inference (ms) of the whole pipeline for each row, mean over the 4 test sets.

For one model it draws three charts. Every panel has a gray line at no defence (pre = none, post = none),
so each chain is always compared with the undefended model.

  pre          the pre-processing chains alone (post = none), grouped by defence set, top 5 shaded
  post         the 15 post-processing chains alone (pre = none), grouped by defence set, top 5 shaded
  combination  the top-5 pre chains x the top-5 post chains (PRE_TOP x POST_TOP), as one defence each

Panels: the metric on the 4 test sets, then "Mean Macro (attacked)" = the mean of the metric over the PGD, Pixle
and PGD + Pixle sets (blue = better than no defence, red = worse, number = change), then Avg Time per Inference.

ConvAE is not a pre-processing defence, so pre chains that contain "convae" are left out (SKIP_PRE).

The 14 pre chains that the final run did not test (post = none, all models) and the top-5 combinations that
it did not test (the 4 other top-5 pre chains x the 5 top-5 post chains, all models) were run later on the
Ronin VM with the same test_all.yaml setup, and added to the tables with merge_missing_pre.py (same folder).
The rows that both runs have gave the same metrics.

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
MODEL_NAMES = {"baseline": "baseline", "robust": "robust", "grl": "GRL",
               "adversarial_retrained": "adversarial retrained", "convae_rn50": "ConvAE + ResNet50"}
ALIASES = {"adv_retrained": "adversarial_retrained"}
NONE = "none"
ATTACKED = ["pgd_test", "pixle_test", "pgdpixle_test"]
MEAN_ATTACKED = "mean_attacked"  # mean of the metric over the ATTACKED test sets
SETS = [("clean_test", "Clean"), ("pgd_test", "PGD"), ("pixle_test", "Pixle"), ("pgdpixle_test", "PGD + Pixle"),
        (MEAN_ATTACKED, "Mean Macro (attacked)")]
PRE_NAMES = ["jpeg", "median", "gaussian"]
POST_NAMES = ["noise", "highconf", "revsig"]
SKIP_PRE = ("convae",)  # ConvAE is not a pre-processing defence
# Top-5 lists for the combination chart (the same lists as in the baseline write-up and in
# src/defenses/testresults_pre_post_processing/combined_baseline/report.md). Pre: highest PGD macro F1.
# Post: highest mean macro F1, ties to the lower clean confidence, duplicates dropped, then
# noise -> revsig in place of highconf -> revsig to show the cost of noise before reverse sigmoid.
PRE_TOP = ["jpeg -> median -> gaussian", "jpeg -> gaussian -> median", "median -> jpeg -> gaussian",
           "median -> gaussian -> jpeg", "gaussian -> median -> jpeg"]
POST_TOP = ["noise", "highconf", "revsig", "noise -> revsig", "highconf -> noise -> revsig"]
METRIC = "f1_macro"
LOWER_IS_BETTER = {"cross_entropy_loss"}
EPS = 1e-9  # values are stored to 6 decimals; any stored difference counts as a change

# Colours (validated reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, RED = "#2a78d6", "#e34948"          # better / worse than no defence
TIME_BAR, TIME_DOT = "#b5b4ae", INK2      # neutral: time is a cost on its own scale
TOP_BAND = "#ebe9e3"                      # neutral shade for the top-5 rows
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
        self.name = MODEL_NAMES.get(tag, tag)  # for the chart titles
        self.metric = metric
        self.sign = -1 if metric in LOWER_IS_BETTER else 1
        self.post = [q for q in chains(POST_NAMES) if (NONE, q) in self.rows]
        kept = [p for p in dict.fromkeys(p for p, _ in self.rows) if p != NONE and not any(s in p for s in SKIP_PRE)]
        # pre chains tested alone (post = none), for the pre chart; in defence-set order when they are jpeg/median/gaussian
        known = [c for c in chains(PRE_NAMES) if c in kept]
        self.pre_alone = known + [p for p in kept if p not in known and (p, NONE) in self.rows]
        # top-5 pre chains tested with every top-5 post chain, for the combination chart
        self.pre_combo = [p for p in PRE_TOP if all((p, q) in self.rows for q in POST_TOP)]

    def value(self, pre, post, s):
        if s == MEAN_ATTACKED:
            return sum(self.value(pre, post, a) for a in ATTACKED) / len(ATTACKED)
        return float(self.rows[(pre, post)][s])

    def ms(self, pre, post):
        """Whole pipeline (pre chain + model + post chain), ms per image."""
        return float(self.times[(pre, post)]["Avg Time per Inference (ms)"])


def set_layout(items, names=POST_NAMES, y=0.0):
    """Chains grouped by defence set: single defences, each pair set, then all three (other chains last).
    Returns rows [(y, chain)], headers [(y, text)] and the next free y."""
    groups = [("single defences", [c for c in items if " -> " not in c])]
    for k in (2, 3):
        sets_ = []
        for c in items:
            parts = c.split(" -> ")
            if len(parts) == k and set(parts) not in sets_:
                sets_.append(set(parts))
        for s in sets_:
            members = [c for c in items if len(c.split(" -> ")) == k and set(c.split(" -> ")) == s]
            title = "{" + ", ".join(n for n in names if n in s) + "}" + (" (all orders)" if k == 3 else "")
            groups.append((title, members))
    grouped = {c for _, m in groups for c in m}
    if any(c not in grouped for c in items):
        groups.append(("other", [c for c in items if c not in grouped]))
    groups = [(t, m) for t, m in groups if m]
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
    ax.set_title(f"{title}\nno defence {base:.3f}", loc="left", fontsize=9,
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
        # The label is always on the right of the point. If the time is lower than no defence,
        # the label starts after the no-defence line, so that it does not cover the bar.
        right = max(v, base)
        if log:
            r = v / base
            txt = f"{v:.1f} ms (×{r:.0f})" if r >= 1.95 else f"{v:.1f} ms (×{r:.1f})"
            x = right * 1.12
        else:
            txt = f"{v:.1f} ms ({v - base:+.1f})"
            x = right + top * 0.03
        ax.text(x, y, txt, va="center", ha="left", fontsize=7, color=INK2)
    ax.set_title(f"Avg Time per Inference\nno defence {base:.1f} ms", loc="left", fontsize=9)


def _figure(n_rows_height):
    fig, axes = plt.subplots(1, len(SETS) + 1, figsize=(22.5, max(0.28 * n_rows_height + 1.9, 3.2)), sharey=True,
                             gridspec_kw={"wspace": 0.2, "width_ratios": [1] * len(SETS) + [1.2]})
    return fig, axes


def _limits(R):
    vals = [R.value(p, q, s) for (p, q) in R.rows if p == NONE or p in R.pre_alone for s, _ in SETS]
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


TIME_NOTE = "Avg Time per Inference (ms): the whole pipeline, mean over the 4 test sets and 3 runs."


def _header(fig, R, title, notes, rings=False):
    """Title, notes and the colour legend above the panels (positions in inches from the top)."""
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.8 / h)
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.45 / h, notes, ha="left", va="top", fontsize=9, color=INK2)
    m = metric_label(R.metric)
    better = "lower" if R.sign < 0 else "higher"
    dot = lambda c, text: plt.Line2D([], [], marker="o", ls="", ms=7, color=c, mec=SURFACE, label=text)
    handles = [plt.Line2D([], [], color=INK2, lw=1.2, label="gray line = no defence"),
               dot(BLUE, f"blue = better than no defence ({better} {m})"),
               dot(RED, "red = worse than no defence"),
               dot(MUTED, "gray = same as no defence")]
    if rings:
        handles.append(plt.Line2D([], [], marker="o", ls="", ms=7, mfc="none", mec=INK2,
                                  label="ring = pre chain alone (post = none)"))
    handles.append(plt.Line2D([], [], ls="", marker="", label=f"+0.123 next to a dot = change in {m} from no defence"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.85 / h), ncol=len(handles),
               fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID, handletextpad=0.6, columnspacing=2.0)


def plot_pre(tag, results_dir=RESULTS, metric=METRIC):
    """The pre-processing chains alone (post = none) vs no defence, grouped by defence set."""
    R = Results(tag, results_dir, metric)
    rows, headers, height = set_layout(R.pre_alone, PRE_NAMES)
    lim = _limits(R)
    with plt.rc_context(STYLE):
        fig, axes = _figure(height)
        for ax, (s, title) in zip(axes, SETS):
            _value_panel(ax, R, title, R.value(NONE, NONE, s), [(y, R.value(c, NONE, s)) for y, c in rows], lim=lim)
        base = R.ms(NONE, NONE)
        _time_panel(axes[-1], base, [(y, R.ms(c, NONE)) for y, c in rows], log=True)
        axes[-1].set_xlabel("Avg Time per Inference (ms, log scale)\npre chain + model")
        _rows_with_top(axes, rows, headers, height, PRE_TOP)
        _header(fig, R, f"Input-level defence: pre-processing chains alone vs no defence, tested on the "
                        f"{R.name} model", GROUP_NOTE)
    return fig


GROUP_NOTE = ("Rows are grouped by defence set, so the same defences in a different order are next to each other. "
              "Shaded rows in bold = the top-5 chains used in the combination chart.\n" + TIME_NOTE)


def _rows_with_top(axes, rows, headers, height, top):
    """Y labels with group headers; the top-5 rows are shaded in every panel and named in bold."""
    for y, c in rows:
        if c in top:
            for ax in axes:
                ax.axhspan(y - 0.5, y + 0.5, color=TOP_BAND, zorder=0, lw=0)
    ticks = [y for y, _ in headers] + [y for y, _ in rows]
    tick_labels = [t for _, t in headers] + [
        f"    {label(c)}  [top 5 #{top.index(c) + 1}]" if c in top else f"    {label(c)}" for _, c in rows]
    _finish(axes, ticks, tick_labels, {y for y, c in rows if c in top}, {y for y, _ in headers}, height,
            [y - 0.7 for y, _ in headers[1:]])


def plot_post(tag, results_dir=RESULTS, metric=METRIC):
    """The 15 post-processing chains alone (pre = none) vs no defence, grouped by defence set."""
    R = Results(tag, results_dir, metric)
    rows, headers, height = set_layout(R.post)
    lim = _limits(R)
    with plt.rc_context(STYLE):
        fig, axes = _figure(height)
        for ax, (s, title) in zip(axes, SETS):
            _value_panel(ax, R, title, R.value(NONE, NONE, s), [(y, R.value(NONE, c, s)) for y, c in rows], lim=lim)
        base = R.ms(NONE, NONE)
        _time_panel(axes[-1], base, [(y, R.ms(NONE, c)) for y, c in rows], log=False)
        axes[-1].set_xlabel("Avg Time per Inference (ms, linear)\nmodel + post chain")
        _rows_with_top(axes, rows, headers, height, POST_TOP)
        _header(fig, R, f"Output-level defence: post-processing chains alone vs no defence, tested on the "
                        f"{R.name} model", GROUP_NOTE)
    return fig


def plot_combined(tag, results_dir=RESULTS, metric=METRIC):
    """The top-5 pre chains + the top-5 post chains as one defence each, grouped by pre chain.
    Only the top-5 pre chains that were tested with all 5 post chains are drawn.
    A hollow ring marks the pre chain alone, so the effect of the post chain is also visible."""
    R = Results(tag, results_dir, metric)
    if not R.pre_combo:
        raise ValueError(f"No top-5 pre chain of '{tag}' was tested with all top-5 post chains.")
    rows, ticks, tick_labels, heads, seps, y = [], [], [], [], [], 0.0
    for a in R.pre_combo:
        if heads:
            seps.append(y - 0.75)
        heads.append(y)
        ticks.append(y)
        tick_labels.append(f"{label(a)} +")
        y += 1
        for b in POST_TOP:
            rows.append((y, a, b))
            ticks.append(y)
            tick_labels.append(f"    {label(b)}")
            y += 1
        y += 0.5
    subheads = set()
    lim = _limits(R)
    with plt.rc_context(STYLE):
        fig, axes = _figure(y)
        for ax, (s, title) in zip(axes, SETS):
            _value_panel(ax, R, title, R.value(NONE, NONE, s), [(yy, R.value(a, b, s)) for yy, a, b in rows],
                         rings=[(yy, R.value(a, NONE, s)) for yy, a, _ in rows], lim=lim)
        base = R.ms(NONE, NONE)
        _time_panel(axes[-1], base, [(yy, R.ms(a, b)) for yy, a, b in rows], log=True)
        axes[-1].set_xlabel("Avg Time per Inference (ms, log scale)\npre chain + model + post chain")
        _finish(axes, ticks, tick_labels, set(heads), subheads, y, seps)
        n = len(rows)
        missing = "" if n == 25 else (f" Only {len(R.pre_combo)} of the 5 top-5 pre chains were tested with the "
                                      "post chains for this model.")
        notes = ("Each combination is one defence: the top-5 pre chain, then the model, then the top-5 post chain. "
                 "Dots and numbers show the combination." + missing + "\n" + TIME_NOTE)
        _header(fig, R, f"Input-level + output-level defence: {n} combinations vs no defence, tested on the "
                        f"{R.name} model", notes, rings=True)
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
