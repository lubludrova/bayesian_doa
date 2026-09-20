import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, PercentFormatter
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reproduce Figure 2 from per-observation results.")
    parser.add_argument("--input", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, default=root / "output")
    args = parser.parse_args()
    source, out = args.input, args.output
    out.mkdir(parents=True, exist_ok=True)
    accuracy = pd.read_csv(source / "per_observation.csv")
    accuracy = accuracy[accuracy.reference_pool == 0]
    cost = pd.read_csv(source / "common_accuracy.csv")
    cost = cost[cost.multiplier == 1.5]
    budgets = np.array([1024, 4096, 16384, 65536, 262144])
    methods = ["physical_rqmc", "quotient_self_normalized"]
    plt.rcParams.update({"font.family": "serif", "font.size": 9.5,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0))
    for col, (k, count) in enumerate(((2, 192), (3, 48))):
        data = accuracy[accuracy.k == k]
        costs = cost[cost.k == k]
        assert len(data) == count * 5 and data.obs.nunique() == count
        assert len(costs) == count and costs.obs.nunique() == count
        assert not data.duplicated(["obs", "n"]).any()
        baseline = data.drop_duplicates("obs").floor
        ax = axes[col]
        ax.axhspan(baseline.quantile(.25), baseline.quantile(.75), color="gray", alpha=.10)
        ax.axhline(baseline.median(), color=".45", ls=":", lw=1.3,
                   label="Reference sampling baseline")
        for method, color, label, marker in zip(
                methods, ("#2166ac", "#e66101"), ("Physical RQMC", "Quotient"), ("o", "s")):
            grouped = data.groupby("n")[method]
            ax.plot(budgets, grouped.median().reindex(budgets), color=color,
                    marker=marker, ms=4, lw=1.5, label=label)
            ax.fill_between(budgets, grouped.quantile(.25).reindex(budgets),
                            grouped.quantile(.75).reindex(budgets), color=color, alpha=.15)
            reached = np.array([(costs[method] <= n).sum() for n in budgets])
            axes[2].plot(budgets, reached / count, color=color,
                         ls="-" if k == 2 else "--", marker=marker,
                         markerfacecolor=color if k == 2 else "white", ms=3, lw=1.3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(rf"({chr(97 + col)}) Accuracy, $K={k}$")
        ax.set_ylabel(r"Empirical joint $W_1$")
    axes[2].set_title("(c) Common accuracy")
    axes[2].set_ylim(.5, 1.03)
    axes[2].set_yticks([.5, .75, 1])
    axes[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[2].set_ylabel("Target reached")
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [rf"$2^{{{int(np.log2(n))}}}$" for n in budgets])
        ax.set_xlabel(r"Likelihood evaluations $N$")
        ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend([handles[i] for i in (1, 2, 0)], [labels[i] for i in (1, 2, 0)],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .90), w_pad=1.1)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"comparison.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
