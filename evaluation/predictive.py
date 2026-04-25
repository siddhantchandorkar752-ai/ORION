"""
ORION — Predictive Evaluation (Phase 4)
=========================================
Computes and visualises:
  - ROC-AUC + PR-AUC
  - Calibration curve (reliability diagram)
  - Brier Score
  - Per-threshold confusion matrix stats

Usage:
    python evaluation/predictive.py \
        --tensor_dir data/processed/tensors \
        --model_path models/temporal/best_model.pt \
        --out_dir evaluation/results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.temporal.risk_model import ORIONRiskModel

plt.style.use("dark_background")
PALETTE = {"accent": "#7c3aed", "pos": "#10b981", "neg": "#ef4444", "neutral": "#64748b"}


def load_model(model_path: str, device: torch.device) -> ORIONRiskModel:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    model = ORIONRiskModel(n_features=meta.get("n_features", 11)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, tensor_dir: Path, split: str, device, batch_size=64):
    d = torch.load(tensor_dir / f"{split}.pt", weights_only=True)
    loader = DataLoader(TensorDataset(d["X"], d["y"]), batch_size=batch_size)
    preds, labels = [], []
    for X, y in loader:
        _, risk = model(X.to(device))
        preds.extend(risk.cpu().numpy())
        labels.extend(y.numpy())
    return np.array(preds), np.array(labels)


def plot_roc(y_true, y_pred, ax, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    ax.plot(fpr, tpr, color=PALETTE["accent"], lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutral"], lw=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["accent"])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    return auc


def plot_pr(y_true, y_pred, ax, title="Precision-Recall Curve"):
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    baseline = y_true.mean()
    ax.plot(rec, prec, color=PALETTE["pos"], lw=2, label=f"AP = {ap:.4f}")
    ax.axhline(baseline, color=PALETTE["neutral"], linestyle="--",
               label=f"Baseline = {baseline:.3f}")
    ax.fill_between(rec, prec, alpha=0.15, color=PALETTE["pos"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    return ap


def plot_calibration(y_true, y_pred, ax, title="Calibration Curve"):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    ax.plot(prob_pred, prob_true, "o-", color=PALETTE["accent"],
            lw=2, label="Model")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutral"], label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()


def plot_risk_distribution(y_true, y_pred, ax, title="Risk Score Distribution"):
    pos_risks = y_pred[y_true == 1]
    neg_risks = y_pred[y_true == 0]
    ax.hist(neg_risks, bins=30, alpha=0.7, color=PALETTE["neutral"],
            label="Survived", density=True)
    ax.hist(pos_risks, bins=30, alpha=0.8, color=PALETTE["neg"],
            label="Deceased", density=True)
    ax.set_xlabel("Predicted Risk Score"); ax.set_ylabel("Density")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()


def main(tensor_dir, model_path, out_dir, split):
    tensor_dir = Path(tensor_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(model_path, device)

    print(f"→ Evaluating on '{split}' split …")
    y_pred, y_true = get_predictions(model, tensor_dir, split, device)

    auc    = roc_auc_score(y_true, y_pred)
    ap     = average_precision_score(y_true, y_pred)
    brier  = brier_score_loss(y_true, y_pred)

    print(f"\n{'─'*40}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  PR-AUC   : {ap:.4f}")
    print(f"  Brier    : {brier:.4f}")
    print(f"{'─'*40}")

    # ── Plot 4-panel evaluation figure ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 12), facecolor="#0f172a")
    fig.suptitle("ORION — Predictive Model Evaluation", fontsize=16,
                 fontweight="bold", color="white", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#1e293b")
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#1e293b")
    ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor("#1e293b")
    ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor("#1e293b")

    plot_roc(y_true, y_pred, ax1)
    plot_pr(y_true, y_pred, ax2)
    plot_calibration(y_true, y_pred, ax3)
    plot_risk_distribution(y_true, y_pred, ax4)

    # Annotate metrics
    fig.text(0.5, 0.01,
             f"ROC-AUC={auc:.4f}  |  PR-AUC={ap:.4f}  |  Brier Score={brier:.4f}",
             ha="center", color="#94a3b8", fontsize=11)

    out_fig = out_dir / "predictive_eval.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Evaluation plot saved → {out_fig}")
    plt.close()

    # Save JSON summary
    metrics = {"roc_auc": auc, "pr_auc": ap, "brier_score": brier, "split": split}
    with open(out_dir / "predictive_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics JSON saved → {out_dir}/predictive_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dir",  default="data/processed/tensors")
    parser.add_argument("--model_path",  default="models/temporal/best_model.pt")
    parser.add_argument("--out_dir",     default="evaluation/results")
    parser.add_argument("--split",       default="test")
    args = parser.parse_args()
    main(args.tensor_dir, args.model_path, args.out_dir, args.split)
