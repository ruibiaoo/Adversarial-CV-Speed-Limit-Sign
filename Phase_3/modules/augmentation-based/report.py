"""Training / validation loss report for the augmentation-based defences.

Reads the per-epoch histories in tuning/augmentation-tuning/ (written by the
training scripts) and writes, in the same folder:
  loss_report.md          per-run tables (best epoch, train / val loss, val acc / F1)
  summary.csv             the same, one row per run
  <defence>_loss.png      train loss and val loss per epoch, one line per setting,
                          with the no-defence baseline as a grey reference

  python modules/augmentation-based/report.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import yaml  # noqa: E402

import _trainer as T  # noqa: E402

DEFENCES = {  # folder -> (title, grid parameter)
    "random_erasing": ("Random erasing", "max_area"),
    "mixup": ("Mixup", "alpha"),
    "trades": ("TRADES", "beta"),
}
# categorical slots 1-3 (validated all-pairs, light surface), baseline in muted ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BASELINE = "#8a8984"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
TRAIN_LOSS_NOTE = {
    "random_erasing": "class-weighted CE on erased images",
    "mixup": "class-weighted CE on mixed images and soft labels",
    "trades": "class-weighted CE + beta * KL(clean || adversarial)",
}


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("epoch", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log")  # early epochs are 10-100x the late ones


def plot(defence, title, runs, baseline, chosen_setting, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, key, sub in ((axes[0], "train_loss", f"Train loss ({TRAIN_LOSS_NOTE[defence]})"),
                         (axes[1], "val_loss", "Val loss (clean val, unweighted CE)")):
        style(ax, sub, "loss (log scale)")
        if baseline:
            ax.plot([r["epoch"] for r in baseline], [r[key] for r in baseline], color=BASELINE,
                    linewidth=1.5, linestyle="--", label="no defence")
        for colour, (setting, hist, best) in zip(SERIES, runs):
            label = setting + (" (chosen)" if setting == chosen_setting else "")
            xs, ys = [r["epoch"] for r in hist], [r[key] for r in hist]
            ax.plot(xs, ys, color=colour, linewidth=2, label=label)
            b = hist[best - 1]
            ax.plot(b["epoch"], b[key], "o", markersize=8, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(setting, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="center")
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(f"{title}: loss per epoch (dot = best checkpoint, by val macro F1)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def row_for(defence, setting, hist, best, chosen):
    b = hist[best - 1]
    lo = min(hist, key=lambda r: r["val_loss"])
    return {"defence": defence, "setting": setting, "epochs_trained": len(hist), "best_epoch": best,
            "train_loss_at_best": round(b["train_loss"], 4), "val_loss_at_best": round(b["val_loss"], 4),
            "min_val_loss": round(lo["val_loss"], 4), "min_val_loss_epoch": int(lo["epoch"]),
            "val_accuracy": round(b["val_accuracy"], 4), "val_macro_f1": round(b["val_macro_f1"], 4),
            "chosen": "yes" if chosen else ""}


def best_epoch(hist):
    return int(max(hist, key=lambda r: r["val_macro_f1"])["epoch"])  # first max, as in training


def main():
    out = T.TUNING
    configs = yaml.safe_load((T.PHASE3 / "defence_configs.yaml").read_text(encoding="utf-8")) or {}
    rows = []

    base_csv = out / "baseline_none" / "history.csv"
    baseline = read(base_csv) if base_csv.exists() else []
    if baseline:
        rows.append(row_for("none (baseline)", "-", baseline, best_epoch(baseline), False))

    for defence, (title, param) in DEFENCES.items():
        grid_csv = out / defence / "grid.csv"
        if not grid_csv.exists():
            continue
        chosen = configs.get(defence, {}).get("params", {}).get(param)
        runs = []
        with grid_csv.open(encoding="utf-8") as f:
            for g in csv.DictReader(f):
                hist = read(out / defence / f"{param}{g[param]}_history.csv")
                setting = f"{param}={g[param]}"
                runs.append((setting, hist, best_epoch(hist)))
                rows.append(row_for(defence, setting, hist, best_epoch(hist),
                                    chosen is not None and float(g[param]) == float(chosen)))
        chosen_setting = f"{param}={chosen}" if chosen is not None else None
        plot(defence, title, runs, baseline, chosen_setting, out / f"{defence}_loss.png")

    T.write_csv(out / "summary.csv", rows)
    write_markdown(out / "loss_report.md", rows, configs)
    print(f"Report written to {out}")


def write_markdown(path, rows, configs):
    cols = ["setting", "epochs_trained", "best_epoch", "train_loss_at_best", "val_loss_at_best",
            "min_val_loss", "min_val_loss_epoch", "val_accuracy", "val_macro_f1", "chosen"]
    lines = [
        "# Augmentation-based defences: training and validation loss",
        "",
        "Generated by `modules/augmentation-based/report.py` from the per-epoch histories in this folder.",
        "",
        "- **Train loss** is each defence's own training objective averaged over the epoch. It includes the",
        "  defence (erased / mixed / adversarial inputs, TRADES' KL term), so it is only comparable",
        "  between settings of the same defence, not across defences.",
        "- **Val loss** is plain cross entropy on clean `val/` images (no class weights, no defence),",
        "  so it is comparable across all runs.",
        "- **Best epoch** is the checkpoint kept (highest clean-val macro F1, as in Phase 1).",
        "  `min_val_loss` shows whether the lowest-loss epoch differs from it.",
        "",
    ]
    groups = {}
    for r in rows:
        groups.setdefault(r["defence"], []).append(r)
    titles = {"none (baseline)": "No defence (baseline)", **{k: v[0] for k, v in DEFENCES.items()}}
    for defence, items in groups.items():
        lines += [f"## {titles.get(defence, defence)}", ""]
        if defence in DEFENCES:
            lines += [f"![{titles[defence]} loss curves]({defence}_loss.png)", ""]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in items]
        lines.append("")
    if "median_smoothing" in configs:
        m = configs["median_smoothing"]
        lines += ["## Median smoothing", "",
                  "Test-time only, nothing is trained, so there are no loss curves. Calibrated on clean val "
                  f"(`median_smoothing/grid.csv`): kernel {m['params']['kernel']}, "
                  f"val accuracy {m['val_accuracy']} (no defence {m['no_defence_val_accuracy']}).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
