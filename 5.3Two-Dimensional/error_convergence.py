#!/usr/bin/env python3
"""
Batch error / convergence-order driver for the two-dimensional reflected BSDE
example solved by 2Dexample.py.

For every (M, N) pair in ``CASES`` the scheme is run once, Y(0, x) is compared
with the pre-computed reference solution, and the discrete L^2 / L^inf errors
together with the empirical convergence orders are reported.

Definitions (identical to error.ipynb):
    L2   = sqrt( mean_i |Y_num(0, x_i) - Y_ref(0, x_i)|^2 )   on the M^2 grid
    Linf = max_i |Y_num(0, x_i) - Y_ref(0, x_i)|
    order p from the least-squares fit  log(error) = a + p * log(dt)

Reference solution: Xref1000.csv / Yref1000.csv in this directory
(M_ref = 360, N_ref = 1000, T = 5, second-order scheme).  Only the t = 0 row
of Yref1000.csv is read, so the 2.5 GB file is never loaded into memory.

Grid alignment: for an exact comparison the numerical grid must be a sub-grid
of the reference grid, i.e. M must divide M_ref = 360 (e.g. M = 45, 60, 72,
90, 120, 180, 360).  For other M (e.g. 64, 128) the reference is bilinearly
interpolated (periodic domain) and the run is marked "bilinear" in the table;
for this reference the interpolation error is about 5e-5 in L2 and 3e-4 in
Linf, negligible while the discretisation error is much larger.

Output: every run writes its own CSV file (by default
error_table_k<kstep>_<timestamp>.csv in this directory), so results of
different runs are never overwritten.  Use --out to choose an explicit path;
a numbered suffix is added if that path already exists.

Usage:
    python error_convergence.py
    python error_convergence.py --cases 120:80,120:160,120:320 --kstep 2
    python error_convergence.py --out my_results.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

T = 5.0                    # terminal time of the reference solution
LB, RB = -np.pi, np.pi     # spatial domain (periodic)

# 1 = first-order Euler / generator approximation, 2 = second-order scheme
KSTEP = 1

# The (M, N) pairs that are run in one go:
#   M = number of space partition points per dimension
#   N = number of time steps
CASES = [
    (120, 220),
    (120, 240),
    (120, 260),
    (120, 280),
    (120, 300),
    (120, 320),
    (120, 340),
    (120, 360),
    (120, 380),
    (120, 400),
    (120, 420),
    (120, 440)
    # add a second spatial resolution to also get an order in dx, e.g.
    # (60, 80), (60, 120), (60, 160), (60, 200),
    # (60, 240), (60, 280), (60, 320), (60, 360),
]

REF_DIR = SCRIPT_DIR
REF_X_FILE = "Xref1000.csv"
REF_Y_FILE = "Yref1000.csv"
REF_T_FILE = "tref1000.csv"      # optional, only used to check T

# How to evaluate the reference on the numerical grid:
#   "exact"    -> require M | M_ref, raise otherwise
#   "bilinear" -> always interpolate (periodic bilinear interpolation)
#   "auto"     -> exact when the grid allows it, bilinear otherwise
REF_INTERPOLATION = "auto"

SHOW_PROGRESS = True
RESULTS_DIR = SCRIPT_DIR           # directory for the result CSV files
RESULTS_PREFIX = "error_table"     # file name: <prefix>_k<kstep>_<timestamp>.csv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def read_first_row(path: Path) -> np.ndarray:
    """Read the first row of a (huge) CSV file with one float per column."""
    with open(path, "r") as f:
        line = f.readline()
    values = np.array(line.split(","), dtype=float)
    if values.size == 0:
        raise ValueError(f"empty file: {path}")
    return values


def load_model_class(script_dir: Path):
    """Import TwoDim_periodic from 2Dexample.py.

    The module name starts with a digit, so a plain ``import`` statement is
    not possible; the file is loaded explicitly instead.
    """
    path = script_dir / "2Dexample.py"
    if not path.is_file():
        raise FileNotFoundError(f"cannot find the solver module: {path}")
    spec = importlib.util.spec_from_file_location("example_2d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TwoDim_periodic


def numbered_path(path: Path) -> Path:
    """Return ``path`` if it does not exist, otherwise ``path_1``, ``path_2``..."""
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def resolve_out_path(out_arg, prefix: str, out_dir, kstep: int):
    """Choose the CSV file of this run; an existing file is never overwritten."""
    if out_arg == "":
        return None
    if out_arg:
        path = Path(out_arg)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = Path(out_dir) / f"{prefix}_k{kstep}_{stamp}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return numbered_path(path)


class ReferenceSolution:
    """Y(t = 0, x) on the uniform periodic grid, read from the reference CSV
    files written by 2Dexample.py."""

    def __init__(self, x_path: Path, y_path: Path, lb: float, rb: float):
        self.lb, self.rb = lb, rb

        X = np.loadtxt(x_path, delimiter=",")
        if X.ndim != 2 or X.shape[0] != 2:
            raise ValueError(
                f"{x_path} should contain 2 rows (x1, x2), got shape {X.shape}"
            )
        m2 = X.shape[1]
        m = int(round(np.sqrt(m2)))
        if m * m != m2:
            raise ValueError(f"{x_path} has {m2} columns, not a square grid")
        self.m = m
        self.dx = (rb - lb) / m

        y = read_first_row(y_path)
        if y.size != m2:
            raise ValueError(
                f"{y_path} has {y.size} columns but the X file has {m2}"
            )
        if not np.all(np.isfinite(y)):
            raise ValueError(f"{y_path}: first row contains non-finite values")

        xg = lb + self.dx * np.arange(m)
        x1 = X[0].reshape(m, m)
        x2 = X[1].reshape(m, m)
        # Convention A (generate_grid): flat index = i2*m + i1, so x1 varies
        # along axis 1 and x2 along axis 0.
        err_a = max(
            np.max(np.abs(x1 - xg[None, :])),
            np.max(np.abs(x2 - xg[:, None])),
        )
        # Convention B: x1 and x2 swapped (any other flatten order is invalid).
        err_b = max(
            np.max(np.abs(x1 - xg[:, None])),
            np.max(np.abs(x2 - xg[None, :])),
        )
        tol = 1e-10 * (rb - lb)
        if err_a <= tol:
            self.field = y.reshape(m, m)         # field[i2, i1]
        elif err_b <= tol:
            self.field = y.reshape(m, m).T       # field[i2, i1]
        else:
            raise ValueError(
                f"{x_path}: x grid is not the expected uniform periodic grid "
                f"(deviation {max(err_a, err_b):.3e})"
            )

    def sample(self, xgrid: np.ndarray, M: int, mode: str = "auto"):
        """Return (Y_ref on ``xgrid``, method string).

        ``xgrid`` is the (M^2, 2) grid of the numerical solution.
        """
        if mode not in ("auto", "exact", "bilinear"):
            raise ValueError(f"unknown interpolation mode: {mode!r}")

        aligned = (self.m % M == 0)
        if aligned:
            k = self.m // M
            xg = self.lb + self.dx * np.arange(self.m)[::k]
            grid_ok = (
                np.max(np.abs(xgrid[:, 0] - np.tile(xg, M))) < 1e-12
                and np.max(np.abs(xgrid[:, 1] - np.repeat(xg, M))) < 1e-12
            )
        else:
            grid_ok = False

        if aligned and grid_ok and mode in ("auto", "exact"):
            # The numerical grid is a sub-grid of the reference grid: pick the
            # corresponding entries (same flat order for both fields).
            return self.field[::k, ::k].ravel(), "exact"

        if mode == "exact":
            divisors = [d for d in range(4, self.m + 1) if self.m % d == 0]
            raise ValueError(
                f"M = {M} does not give a sub-grid of the reference grid "
                f"(M_ref = {self.m}); use one of {divisors} "
                f"or set REF_INTERPOLATION = 'auto'/'bilinear'"
            )

        return self._bilinear(xgrid[:, 0], xgrid[:, 1]), "bilinear"

    def _bilinear(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Periodic bilinear interpolation of ``self.field`` at (x1, x2)."""
        m = self.m
        u1 = (x1 - self.lb) / self.dx
        u2 = (x2 - self.lb) / self.dx
        f1, i1 = np.modf(u1)
        f2, i2 = np.modf(u2)
        i1 = i1.astype(np.int64) % m
        i2 = i2.astype(np.int64) % m
        i1p = (i1 + 1) % m
        i2p = (i2 + 1) % m
        F = self.field
        return (
            (1.0 - f2) * (1.0 - f1) * F[i2, i1]
            + (1.0 - f2) * f1 * F[i2, i1p]
            + f2 * (1.0 - f1) * F[i2p, i1]
            + f2 * f1 * F[i2p, i1p]
        )


def solve_y0(model, kstep: int, show_progress: bool = True, tag: str = ""):
    """Backward sweep of the scheme, returning only Y(t = 0, .).

    This reproduces ``TwoDim_periodic.exGA_solver`` step by step but keeps only
    the ``kstep + 1`` most recent time levels instead of all N + 1 levels,
    which for e.g. M = 360, N = 1000 would need ~4 GB.
    """
    if kstep not in (1, 2):
        raise ValueError("kstep must be 1 or 2")
    N = model.N
    if N < kstep:
        raise ValueError(f"N = {N} is too small for kstep = {kstep}")

    yN = model.terminal(model.tgrid[N], model.xgrid)
    if kstep == 1:
        future = [yN]                              # Y_{n+1}
        n_start = N - 1
    else:
        # same initialisation as exGA_solver: one Euler step for Y_{N-1}
        future = [model.Euler_step(N - 1, yN)[0], yN]   # Y_{n+1}, Y_{n+2}
        n_start = N - 2
    if n_start < 0:
        return future[0]

    total = n_start + 1
    every = max(1, total // 5)
    for n in range(n_start, -1, -1):
        known = np.stack(future, axis=1)           # (M^2, kstep)
        future = [model.exGA_step(n, kstep, known)[0]] + future[: kstep - 1]

        done = n_start - n + 1
        if show_progress and (done % every == 0 or n == 0):
            msg = f"    {tag} {100.0 * done / total:3.0f}% ({done}/{total})"
            if sys.stderr.isatty():
                print("\r" + msg, end="", file=sys.stderr, flush=True)
            else:
                print(msg, file=sys.stderr, flush=True)
    if show_progress and sys.stderr.isatty():
        print(file=sys.stderr)

    return future[0]


def loglog_slope(x, err) -> float:
    """Least-squares slope of log(err) against log(x) (the convergence order)."""
    x = np.asarray(x, dtype=float)
    err = np.asarray(err, dtype=float)
    ok = np.isfinite(x) & np.isfinite(err) & (x > 0) & (err > 0)
    if ok.sum() < 2:
        return float("nan")
    A = np.column_stack((np.ones(ok.sum()), np.log(x[ok])))
    coef, *_ = np.linalg.lstsq(A, np.log(err[ok]), rcond=None)
    return float(coef[1])


def pair_rate(prev: dict, cur: dict, which: str) -> float:
    """Order between two consecutive runs.

    ``which`` = "L2" or "Linf".  The order is computed with respect to dt when
    M is unchanged, and with respect to dx when N is unchanged; otherwise the
    two errors cannot be compared and NaN is returned.
    """
    # if prev["M"] == cur["M"] and prev["N"] != cur["N"]:
    #     x0, x1 = prev["dt"], cur["dt"]
    # elif prev["N"] == cur["N"] and prev["M"] != cur["M"]:
    #     x0, x1 = prev["dx"], cur["dx"]
    # else:
    #     return float("nan")
    x0, x1 = prev["dt"], cur["dt"]
    e0, e1 = prev[which], cur[which]
    if min(e0, e1) <= 0 or x0 == x1:
        return float("nan")
    return float(np.log(e1 / e0) / np.log(x1 / x0))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def preflight(cases, model_cls, ref, kstep, mode):
    """Validate every case before the (possibly long) sweep starts."""
    for M, N in cases:
        if M < 4:
            raise ValueError(f"M = {M} is too small (M >= 4 required)")
        if N < kstep:
            raise ValueError(f"N = {N} is too small for kstep = {kstep}")
        model = model_cls(lb=LB, rb=RB, N=N, M=M, T=T)
        ref.sample(model.xgrid, M, mode=mode)


def run_cases(cases, model_cls, ref, kstep, mode, show_progress, out_path):
    results = []
    writer = None
    fh = None
    if out_path is not None:
        fh = open(out_path, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(
            ["M", "N", "dt", "L2", "Linf", "rate_L2", "rate_Linf",
             "ref_grid", "kstep", "time_s"]
        )

    header = (
        f"{'M':>5} {'N':>5} {'dt':>10} {'L2 error':>14} {'Linf error':>14} "
        f"{'rate L2':>9} {'rate Linf':>10} {'ref grid':>9} {'time[s]':>9}"
    )
    print(header)
    print("-" * len(header))

    def fmt_rate(v):
        return f"{v:9.3f}" if np.isfinite(v) else f"{'-':>9}"

    warned = set()
    try:
        for i, (M, N) in enumerate(cases, start=1):
            if M < 4:
                raise ValueError(f"M = {M} is too small")
            model = model_cls(lb=LB, rb=RB, N=N, M=M, T=T)
            model.kstep = kstep
            model.alpha = model.get_alpha(kstep)
            model.beta = model.get_beta(kstep)

            # sample the reference before the (long) solve so that a bad grid
            # configuration fails immediately
            yref, method = ref.sample(model.xgrid, M, mode=mode)
            if method == "bilinear" and M not in warned:
                warned.add(M)
                print(
                    f"      note: M = {M} does not divide M_ref = {ref.m}; "
                    f"the reference is bilinearly interpolated",
                    file=sys.stderr,
                )

            tag = f"[{i}/{len(cases)}] M={M} N={N}"
            tic = time.perf_counter()
            # the solver prints one line per time step; silence it
            with redirect_stdout(io.StringIO()):
                y0 = solve_y0(model, kstep, show_progress, tag=tag)
            elapsed = time.perf_counter() - tic

            diff = y0 - yref
            res = {
                "M": M,
                "N": N,
                "dt": T / N,
                "dx": (RB - LB) / M,
                "L2": float(np.sqrt(np.mean(diff ** 2))),
                "Linf": float(np.max(np.abs(diff))),
                "ref_grid": method,
                "time_s": elapsed,
            }

            if results:
                res["rate_L2"] = pair_rate(results[-1], res, "L2")
                res["rate_Linf"] = pair_rate(results[-1], res, "Linf")
            else:
                res["rate_L2"] = res["rate_Linf"] = float("nan")
            results.append(res)

            print(
                f"{M:5d} {N:5d} {T / N:10.4e} {res['L2']:14.6e} "
                f"{res['Linf']:14.6e} {fmt_rate(res['rate_L2'])} "
                f"{fmt_rate(res['rate_Linf'])} {method:>9} {elapsed:9.2f}"
            )

            if writer is not None:
                writer.writerow(
                    [M, N, res["dt"], res["L2"], res["Linf"],
                     f"{res['rate_L2']:.6f}" if np.isfinite(res["rate_L2"]) else "",
                     f"{res['rate_Linf']:.6f}" if np.isfinite(res["rate_Linf"]) else "",
                     method, kstep, f"{elapsed:.3f}"]
                )
                fh.flush()
    finally:
        if fh is not None:
            fh.close()

    return results


def print_orders(results):
    if len(results) < 2:
        return
    print()
    print("Convergence orders from log-log least-squares fits")
    print("  (error ~ dt^p; for dx-refinement the same fit is used with dx):")

    def line(label, rows, xkey):
        l2 = loglog_slope([r[xkey] for r in rows], [r["L2"] for r in rows])
        li = loglog_slope([r[xkey] for r in rows], [r["Linf"] for r in rows])
        print(f"    {label:<32} ({len(rows):2d} cases)  "
              f"L2: {l2:6.3f}   Linf: {li:6.3f}")

    # mixed fit, like error.ipynb (only meaningful if more than one M is used)
    if len({r["M"] for r in results}) > 1:
        line("all cases vs dt (mixed)", results, "dt")

    # temporal orders: fixed M, varying N
    for M in sorted({r["M"] for r in results}):
        rows = [r for r in results if r["M"] == M]
        if len({r["N"] for r in rows}) >= 2:
            line(f"M = {M}, refine N", rows, "dt")

    # spatial orders: fixed N, varying M
    for N in sorted({r["N"] for r in results}):
        rows = [r for r in results if r["N"] == N]
        if len({r["M"] for r in rows}) >= 2:
            line(f"N = {N}, refine M", rows, "dx")


def parse_cases(text: str):
    cases = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.replace("x", ":").replace(" ", ":").split(":")
        parts = [p for p in parts if p]
        if len(parts) != 2:
            raise ValueError(f"cannot parse case {token!r} (expected M:N)")
        cases.append((int(parts[0]), int(parts[1])))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="L2/Linf errors and convergence orders for 2Dexample.py"
    )
    ap.add_argument(
        "--cases",
        help='override CASES, e.g. "120:80,120:160,120:320"',
    )
    ap.add_argument("--kstep", type=int, choices=(1, 2), default=KSTEP)
    ap.add_argument(
        "--interp", choices=("auto", "exact", "bilinear"),
        default=REF_INTERPOLATION,
        help="how to put the reference solution on the numerical grid",
    )
    ap.add_argument(
        "--out", default=None,
        help="exact CSV output path (default: a new timestamped file in "
             "RESULTS_DIR; if the path already exists a numbered suffix is "
             "added, so nothing is overwritten; empty string: do not write)",
    )
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    cases = parse_cases(args.cases) if args.cases else list(CASES)
    if not cases:
        ap.error("the case list is empty")

    ref_dir = Path(REF_DIR)
    x_path = ref_dir / REF_X_FILE
    y_path = ref_dir / REF_Y_FILE
    for p in (x_path, y_path):
        if not p.is_file():
            raise FileNotFoundError(f"reference file not found: {p}")

    t_path = ref_dir / REF_T_FILE
    if t_path.is_file():
        t_ref = np.loadtxt(t_path, delimiter=",")
        if not np.isclose(t_ref[-1], T):
            raise ValueError(
                f"T = {T} but the reference time grid ends at {t_ref[-1]}; "
                f"use T = {t_ref[-1]} or a matching reference"
            )

    print(f"reference: {x_path.name}, {y_path.name}")
    print(f"T = {T}, kstep = {args.kstep}, cases = {len(cases)}")
    ref = ReferenceSolution(x_path, y_path, LB, RB)
    print(f"reference grid: M_ref = {ref.m}")
    model_cls = load_model_class(SCRIPT_DIR)
    preflight(cases, model_cls, ref, args.kstep, args.interp)

    out_path = resolve_out_path(args.out, RESULTS_PREFIX, RESULTS_DIR, args.kstep)
    if out_path is not None:
        print(f"results file: {out_path}")

    results = run_cases(
        cases, model_cls, ref, args.kstep, args.interp,
        SHOW_PROGRESS and not args.no_progress, out_path,
    )

    print_orders(results)

    if out_path is not None:
        print(f"\nresults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
