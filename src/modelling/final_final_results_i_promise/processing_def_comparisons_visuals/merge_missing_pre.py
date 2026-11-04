"""Add the rows of a later test_all.py run to the tables in final_final_results_i_promise/.

The later runs (configs in missing_pre_run/ on the Ronin VM) used the same test_all.yaml settings,
test sets, models and timing as the final run, but tested rows that the final run did not test:
  test_all_missing_pre.yaml          the 14 pre-processing chains, post = none, all models
  test_all_top5_combos.yaml          the 4 other top-5 pre chains x the 5 top-5 post chains, baseline
  test_all_top5_combos_others.yaml   the same 20 combinations for the 4 other models
This script:

  1. checks the rows that both runs have (for example none and jpeg -> median -> gaussian, post = none):
     the metrics must be the same, so the two runs are comparable;
  2. adds only the rows that are not yet in the tables, below the rows of the same model,
     in every <metric>.txt and in runtime.txt (same table format as test_all.py);
  3. copies the tests/<combo>/ folders of the added rows.

Rows that are already in the tables do not change.

  python merge_missing_pre.py <new run folder>            # check and merge
  python merge_missing_pre.py <new run folder> --check    # check only, change nothing
"""
import argparse
import re
import shutil
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]  # final_final_results_i_promise
KEY = 3  # Preprocessing, Model, Postprocessing


def read(path):
    """Fixed-width table -> (title line, headers, rows as lists of cells)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    dash = next(i for i, ln in enumerate(lines) if ln.startswith("---"))
    headers = re.split(r"\s{2,}", lines[dash - 1].strip())
    rows = [re.split(r"\s{2,}", ln.strip()) for ln in lines[dash + 1:] if ln.strip()]
    assert all(len(r) == len(headers) for r in rows), f"bad row in {path}"
    return lines[0], headers, rows


def format_table(headers, rows, left_align_cols=3):
    """Same as test_all._format_table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    def fmt(cells):
        return "  ".join(str(c).ljust(widths[i]) if i < left_align_cols else str(c).rjust(widths[i])
                         for i, c in enumerate(cells))
    return "\n".join([fmt(headers), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows]) + "\n"


def folder(pre, model, post):
    tok = lambda s: "none" if s == "none" else "_then_".join(s.split(" -> "))
    return f"pre_{tok(pre)}__model_{model}__post_{tok(post)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("new", help="output folder of the later run (has <metric>.txt, runtime.txt, tests/)")
    ap.add_argument("--check", action="store_true", help="only compare the shared rows")
    args = ap.parse_args()
    new_dir = Path(args.new)

    files = sorted(p.name for p in new_dir.glob("*.txt"))
    assert "f1_macro.txt" in files and "runtime.txt" in files, f"no tables in {new_dir}"
    missing = [f for f in files if not (FINAL / f).exists()]
    assert not missing, f"not in {FINAL}: {missing}"

    # 1. shared rows must agree on every metric (runtime can differ from run to run)
    bad = 0
    for name in files:
        if name == "runtime.txt":
            continue
        _, h_old, old = read(FINAL / name)
        _, h_new, new = read(new_dir / name)
        assert h_old == h_new, f"{name}: columns differ {h_old} vs {h_new}"
        old_by_key = {tuple(r[:KEY]): r for r in old}
        for r in new:
            o = old_by_key.get(tuple(r[:KEY]))
            if o is not None and o != r:
                bad += 1
                print(f"DIFFERENT {name}: {r[:KEY]}\n  final {o[KEY:]}\n  new   {r[KEY:]}")
    _, _, old_rt = read(FINAL / "runtime.txt")
    _, _, new_rt = read(new_dir / "runtime.txt")
    old_rt = {tuple(r[:KEY]): r for r in old_rt}
    print("Shared rows, time per image (ms): final run vs new run")
    for r in new_rt:
        o = old_rt.get(tuple(r[:KEY]))
        if o is not None:
            print(f"  {' | '.join(r[:KEY]):45s} {o[-1]:>10s} {r[-1]:>10s}")
    print("Shared rows: all metrics are the same." if not bad else f"Shared rows: {bad} metric rows differ.")
    if args.check:
        return
    if bad:
        raise SystemExit("Not merged: the shared rows differ, so the runs are not comparable.")

    # 2. add the new rows below the rows of the same model
    added = None
    for name in files:
        title, headers, old = read(FINAL / name)
        _, _, new = read(new_dir / name)
        have = {tuple(r[:KEY]) for r in old}
        extra = [r for r in new if tuple(r[:KEY]) not in have]
        out = list(old)
        for r in extra:
            last = max((i for i, o in enumerate(out) if o[1] == r[1]), default=len(out) - 1)
            out.insert(last + 1, r)
        (FINAL / name).write_text(f"{title}\n\n" + format_table(headers, out), encoding="utf-8")
        keys = [tuple(r[:KEY]) for r in extra]
        assert added is None or keys == added, f"{name}: other new rows than the other tables"
        added = keys
    print(f"Added {len(added)} rows to each of {len(files)} tables.")

    # 3. copy the tests/<combo>/ folders of the added rows
    for pre, model, post in added:
        src = new_dir / "tests" / folder(pre, model, post)
        dst = FINAL / "tests" / src.name
        assert src.is_dir(), f"missing {src}"
        assert not dst.exists(), f"already there: {dst}"
        shutil.copytree(src, dst)
    print(f"Copied {len(added)} folders to {FINAL / 'tests'}.")


if __name__ == "__main__":
    main()
