"""
ORION — Temporal Risk Model (LSTM)
====================================
Architecture:
  Input:  [batch, seq_len, n_features]  (24h × 11 vitals/labs)
  LSTM:   2-layer bidirectional LSTM
  Head:   FC → Dropout → sigmoid → risk score ∈ [0,1]

Outputs:
  - risk score at each time step (for UI timeline)
  - final risk at horizon (for causal engine input)
"""

import torch
import torch.nn as nn


class ORIONRiskModel(nn.Module):
    """
    Bidirectional 2-layer LSTM for ICU mortality / sepsis risk scoring.
    Returns per-step risk scores AND a final aggregated risk.
    """

    def __init__(
        self,
        n_features: int = 11,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Per-time-step risk head
        self.step_head = nn.Sequential(
            nn.Linear(hidden_size * D, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Final risk (uses last hidden state attention-pooled)
        self.attn = nn.Linear(hidden_size * D, 1)
        self.final_head = nn.Sequential(
            nn.Linear(hidden_size * D, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, F]
        Returns:
            step_risks:  [B, T]   — risk at each hour
            final_risk:  [B]      — aggregated risk score
        """
        out, _ = self.lstm(x)                       # [B, T, H*D]

        # Per-step risk
        step_risks = self.step_head(out).squeeze(-1) # [B, T]

        # Attention pooling over time
        attn_w = torch.softmax(self.attn(out), dim=1)  # [B, T, 1]
        context = (attn_w * out).sum(dim=1)             # [B, H*D]
        final_risk = self.final_head(context).squeeze(-1)  # [B]

        return step_risks, final_risk


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance (ICU mortality is rare).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt  = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


if __name__ == "__main__":
    # Smoke test
    model = ORIONRiskModel(n_features=11)
    x = torch.randn(4, 24, 11)
    step_risks, final_risk = model(x)
    print(f"step_risks: {step_risks.shape}")   # [4, 24]
    print(f"final_risk: {final_risk.shape}")   # [4]
    print("✓ Model forward pass OK")
