import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd


BUDGETS = np.array([2**i for i in (10, 12, 14, 16, 18, 20, 22)])
METHODS = (
    ("physical_rqmc", "#4C78A8", "Physical RQMC", "o"),
    ("quotient_self_normalized", "#E7965C", "Quotient", "s"),
)
COUNTS = {2: 192, 3: 48, 4: 48}
K_STYLES = {2: "-", 3: (0, (5, 2)), 4: (0, (3, 1, 1, 1))}
K_MARKERS = {2: "o", 3: "^", 4: "D"}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reproduce the four-panel accuracy figure.")
    parser.add_argument("--input", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, default=root / "output")
    parser.add_argument("--draws", type=int, choices=(100, 500), default=500)
    parser.add_argument("--cost-draws", type=int, choices=(100, 500), default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    accuracy = pd.read_csv(args.input / "per_observation.csv")
    accuracy = accuracy[(accuracy.m == args.draws) & (accuracy.reference_pool == 0)]
    costs = pd.read_csv(args.input / "common_accuracy.csv")
    costs = costs[(costs.m == args.cost_draws) & (costs.reference_pool == 0)
                  & (costs.multiplier == 1.5)]
    for sources, count in COUNTS.items():
        assert len(accuracy[accuracy.k == sources]) == count * len(BUDGETS)
        assert len(costs[costs.k == sources]) == count

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9.0, "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    figure, axes = plt.subplots(1, 4, figsize=(7.16, 1.78),
                                gridspec_kw={"width_ratios": (1, 1, 1, 1.08)})
    for panel, (axis, sources) in enumerate(zip(axes[:3], COUNTS)):
        data = accuracy[accuracy.k == sources]
        baseline = data.drop_duplicates("obs").floor
        axis.axhspan(baseline.quantile(.25), baseline.quantile(.75),
                     color="gray", alpha=.10)
        axis.axhline(baseline.median(), color=".48", ls=":", lw=.9)
        for method, color, _, marker in METHODS:
            grouped = data.groupby("n")[method]
            axis.plot(BUDGETS, grouped.median().reindex(BUDGETS), color=color,
                      marker=marker,
                      markerfacecolor="white" if method == METHODS[0][0] else color,
                      markeredgecolor=color, markeredgewidth=.7, ms=2.8, lw=1.0)
            axis.fill_between(BUDGETS, grouped.quantile(.25).reindex(BUDGETS),
                              grouped.quantile(.75).reindex(BUDGETS),
                              color=color, alpha=.12)
        axis.set_title(rf"({chr(97 + panel)}) Accuracy, $K={sources}$", fontsize=8.8, pad=3)
        axis.set_yscale("log")
        axis.set_ylim(.015, .2)
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        if panel == 0:
            axis.set_ylabel(r"Joint $W_1$", labelpad=2)
        else:
            axis.tick_params(labelleft=False)

    coverage = axes[3]
    for sources, count in COUNTS.items():
        group = costs[costs.k == sources]
        for method, color, _, _ in METHODS:
            reached = np.array([(group[method] <= budget).sum() for budget in BUDGETS])
            coverage.plot(BUDGETS, reached / count, color=color, ls=K_STYLES[sources],
                          marker=K_MARKERS[sources],
                          markerfacecolor="white" if method == METHODS[0][0] else color,
                          markeredgecolor=color, markeredgewidth=.7, ms=3.0, lw=1.0)
    coverage.set_title("(d) Target coverage", fontsize=9, pad=2)
    coverage.set_ylim(.45, 1.03)
    coverage.set_yticks([.5, .75, 1.0])
    coverage.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "1.0" if np.isclose(value, 1.0) else f"{value:g}")
    )
    coverage.yaxis.tick_right()
    coverage.yaxis.set_label_position("right")
    coverage.set_ylabel("Fraction reaching target", labelpad=5)
    coverage.spines["left"].set_visible(False)
    coverage.spines["right"].set_visible(True)

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS, [rf"$2^{{{int(np.log2(n))}}}$" if i % 2 == 0 else ""
                                  for i, n in enumerate(BUDGETS)])
        axis.tick_params(axis="both", labelsize=8.3, length=2.8, width=.7, pad=4.0)
        axis.grid(alpha=.14, linewidth=.5)

    method_handles = [Line2D([0], [0], color=color, marker=marker,
                             markerfacecolor="white" if method == METHODS[0][0] else color,
                             ms=3, lw=1) for method, color, _, marker in METHODS]
    k_handles = [Line2D([0], [0], color="black", ls=K_STYLES[k], marker=K_MARKERS[k],
                        markerfacecolor="white", ms=3, lw=1) for k in COUNTS]
    figure.legend([*method_handles, Line2D([0], [0], color=".48", ls=":", lw=.9)],
                  [METHODS[0][2], METHODS[1][2], "Sampling baseline"],
                  loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.36, 1.01),
                  fontsize=8.5, handlelength=1.4, handletextpad=.3, columnspacing=.8)
    figure.legend(k_handles, ["$K=2$", "$K=3$", "$K=4$"],
                  loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.77, 1.01),
                  fontsize=8.5, handlelength=1.4, handletextpad=.3, columnspacing=.75)
    figure.supxlabel(r"Likelihood evaluations $N$", fontsize=9, y=.03)
    figure.subplots_adjust(left=.065, right=.91, bottom=.25, top=.72, wspace=.16)
    figure.savefig(args.output / "comparison.pdf", bbox_inches="tight", pad_inches=.02)
    figure.savefig(args.output / "comparison.png", dpi=240,
                   bbox_inches="tight", pad_inches=.02)
    plt.close(figure)


if __name__ == "__main__":
    main()
