"""
ORION — LSTM Training Script
==============================
Trains ORIONRiskModel on preprocessed MIMIC-IV tensors.
Handles class imbalance, early stopping, and checkpointing.

Usage:
    python models/temporal/train.py --tensor_dir data/processed/tensors \
                                     --out_dir models/temporal \
                                     --epochs 50 --lr 1e-3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.temporal.risk_model import ORIONRiskModel, FocalLoss


def load_split(tensor_dir: Path, split: str):
    d = torch.load(tensor_dir / f"{split}.pt", weights_only=True)
    return d["X"], d["y"]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        _, final_risk = model(X_batch)
        loss = criterion(final_risk, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        _, final_risk = model(X_batch)
        all_preds.extend(final_risk.cpu().numpy())
        all_labels.extend(y_batch.numpy())
    auc = roc_auc_score(all_labels, all_preds)
    return auc


def main(tensor_dir, out_dir, epochs, lr, batch_size, patience):
    tensor_dir = Path(tensor_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(tensor_dir / "meta.json") as f:
        meta = json.load(f)
    n_features = meta["n_features"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"→ Device: {device}  |  Features: {n_features}")

    # Load splits
    X_train, y_train = load_split(tensor_dir, "train")
    X_val,   y_val   = load_split(tensor_dir, "val")
    X_test,  y_test  = load_split(tensor_dir, "test")

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),
                              batch_size=batch_size)

    model = ORIONRiskModel(n_features=n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, verbose=True)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_auc   = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val AUC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": val_auc,
                "meta": meta,
            }, out_dir / "best_model.pt")
            print(f"  ✓ Saved best model (AUC={val_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test evaluation
    ckpt = torch.load(out_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_auc = evaluate(model, test_loader, device)
    print(f"\n✅ Test AUC: {test_auc:.4f}  (best val AUC: {best_auc:.4f})")

    results = {"test_auc": test_auc, "best_val_auc": best_auc,
               "epochs_trained": epoch}
    with open(out_dir / "train_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_dir}/train_results.json")
    print("\nNext: python models/causal/causal_engine.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dir", default="data/processed/tensors")
    parser.add_argument("--out_dir",    default="models/temporal")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience",   type=int, default=10)
    args = parser.parse_args()
    main(args.tensor_dir, args.out_dir, args.epochs,
         args.lr, args.batch_size, args.patience)
