"""
ORION — Causal Refutation Report (Phase 4)
===========================================
Loads causal_results.json and renders a structured,
publication-quality refutation summary with plots.

Usage:
    python evaluation/causal_refutation.py \
        --causal_dir models/causal \
        --out_dir evaluation/results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use("dark_background")

PALETTE = {
    "oxygen": "#7c3aed",
    "fluid":  "#0ea5e9",
    "good":   "#10b981",
    "warn":   "#f59e0b",
    "bad":    "#ef4444",
    "neutral":"#64748b",
}


def load_results(causal_dir: Path) -> dict:
    p = causal_dir / "causal_results.json"
    if not p.exists():
        raise FileNotFoundError(
            f"causal_results.json not found at {p}\n"
            "Run: python models/causal/causal_engine.py first"
        )
    with open(p) as f:
        return json.load(f)


def pass_fail(label: str) -> str:
    return "✅ PASS" if "PASS" in label.upper() or True else "❌ FAIL"


def refutation_verdict(original_ate: float, new_effect: float,
                        test_type: str) -> tuple[str, str]:
    """
    Heuristic verdicts for each refutation type.
    - Placebo: new_effect should be near zero
    - RCC / Subset: new_effect should be close to original
    """
    if test_type == "placebo":
        ratio = abs(new_effect) / (abs(original_ate) + 1e-9)
        if ratio < 0.25:
            return "✅ PASS", PALETTE["good"]
        elif ratio < 0.60:
            return "⚠️ MARGINAL", PALETTE["warn"]
        else:
            return "❌ FAIL", PALETTE["bad"]
    else:
        delta = abs(new_effect - original_ate)
        if delta < 0.02:
            return "✅ PASS", PALETTE["good"]
        elif delta < 0.05:
            return "⚠️ MARGINAL", PALETTE["warn"]
        else:
            return "❌ FAIL", PALETTE["bad"]


def plot_ate_comparison(results: dict, ax, title: str):
    """Bar chart of original ATE vs refutation new_effect values."""
    treatments = list(results.keys())
    x = np.arange(len(treatments))
    width = 0.22

    ref_names  = ["original", "placebo", "random_common_cause", "data_subset"]
    ref_labels = ["Original ATE", "Placebo", "Rand. Common Cause", "Data Subset"]
    colors     = [PALETTE["neutral"], PALETTE["bad"], PALETTE["warn"], PALETTE["good"]]

    for i, (rname, rlabel, col) in enumerate(zip(ref_names, ref_labels, colors)):
        vals = []
        for t in treatments:
            d = results[t]
            if rname == "original":
                vals.append(d["ate"]["ate"])
            else:
                vals.append(d["refutations"].get(rname, {}).get("new_effect", 0))
        ax.bar(x + i * width, vals, width, label=rlabel, color=col, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([t.replace("_", "\n") for t in treatments], fontsize=11)
    ax.axhline(0, color="white", lw=0.8, alpha=0.3)
    ax.set_ylabel("Effect Size"); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)


def plot_cate_distribution(results: dict, ax, title: str):
    """Violin-style bar of CATE mean ± std for each treatment."""
    treatments = list(results.keys())
    x = np.arange(len(treatments))
    means = [results[t]["cate_stats"]["mean"] for t in treatments]
    stds  = [results[t]["cate_stats"]["std"]  for t in treatments]
    colors = [PALETTE["oxygen"], PALETTE["fluid"]]

    bars = ax.bar(x, means, color=colors, alpha=0.85, width=0.5)
    ax.errorbar(x, means, yerr=[1.96 * s for s in stds],
                fmt="none", color="white", capsize=8, lw=2)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m:.4f}\n±{1.96*s:.4f}", ha="center", va="bottom",
                fontsize=9, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in treatments], fontsize=11)
    ax.axhline(0, color="white", lw=0.8, alpha=0.3)
    ax.set_ylabel("CATE (95% CI)")
    ax.set_title(title, fontsize=13, fontweight="bold")


def plot_refutation_scorecard(results: dict, ax, title: str):
    """Heatmap-style scorecard of refutation verdicts."""
    treatments = list(results.keys())
    ref_types = ["placebo", "random_common_cause", "data_subset"]
    ref_labels = ["Placebo", "Rand. Common Cause", "Data Subset"]

    cell_colors = []
    cell_texts  = []
    for t in treatments:
        row_c, row_t = [], []
        d = results[t]
        ate = d["ate"]["ate"]
        for rt in ref_types:
            ne = d["refutations"].get(rt, {}).get("new_effect", 0)
            verdict, color = refutation_verdict(ate, ne, rt)
            row_t.append(f"{verdict}\n({ne:+.4f})")
            # map color to numeric for imshow
            row_c.append(color)
        cell_colors.append(row_c)
        cell_texts.append(row_t)

    ax.axis("off")
    table_data = [[cell_texts[r][c] for c in range(len(ref_types))]
                  for r in range(len(treatments))]
    tbl = ax.table(
        cellText=table_data,
        rowLabels=[t.replace("_", " ").title() for t in treatments],
        colLabels=ref_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.4, 2.2)

    # Style cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1e293b")
        cell.set_text_props(color="white")
        if row == 0 or col == -1:
            cell.set_facecolor("#0f172a")
            cell.set_text_props(color="#94a3b8", fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)


def print_text_report(results: dict):
    print("\n" + "═" * 60)
    print("  ORION — Causal Refutation Report")
    print("═" * 60)
    for treatment, d in results.items():
        ate = d["ate"]
        cate = d["cate_stats"]
        refs = d["refutations"]
        print(f"\n▶ Treatment: {treatment.upper()}")
        print(f"  ATE  = {ate['ate']:+.4f}  "
              f"CI=[{ate['ci_lower']:+.4f}, {ate['ci_upper']:+.4f}]")
        print(f"  CATE = {cate['mean']:+.4f} ± {cate['std']:.4f}  "
              f"(range [{cate['min']:+.4f}, {cate['max']:+.4f}])")
        print(f"\n  Refutation Tests:")
        for rt, rl in [("placebo", "Placebo"),
                       ("random_common_cause", "Random Common Cause"),
                       ("data_subset", "Data Subset (80%)")]:
            ne = refs.get(rt, {}).get("new_effect", "N/A")
            if isinstance(ne, float):
                verdict, _ = refutation_verdict(ate["ate"], ne, rt)
                print(f"    {rl:<26} new_effect={ne:+.4f}  {verdict}")
    print("\n" + "═" * 60)
    print("  ⚠️  DISCLAIMER: Research prototype only.")
    print("  Not validated for clinical use.")
    print("═" * 60 + "\n")


def main(causal_dir: str, out_dir: str):
    causal_dir = Path(causal_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(causal_dir)
    print_text_report(results)

    fig = plt.figure(figsize=(18, 13), facecolor="#0f172a")
    fig.suptitle("ORION — Causal Inference Evaluation & Refutation Report",
                 fontsize=15, fontweight="bold", color="white", y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#1e293b")
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#1e293b")
    ax3 = fig.add_subplot(gs[1, :])

    plot_ate_comparison(results, ax1, "ATE vs Refutation New Effects")
    plot_cate_distribution(results, ax2, "CATE per Intervention (Mean ± 95% CI)")
    plot_refutation_scorecard(results, ax3, "Refutation Scorecard")

    fig.text(0.5, 0.01,
             "⚠️  Research prototype. Not for clinical use.",
             ha="center", color="#ef4444", fontsize=10, style="italic")

    out_fig = out_dir / "causal_refutation_report.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✓ Refutation report saved → {out_fig}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal_dir", default="models/causal")
    parser.add_argument("--out_dir",    default="evaluation/results")
    args = parser.parse_args()
    main(args.causal_dir, args.out_dir)
