"""
ORION — Counterfactual Simulator (Phase 3)
==========================================
Core engine that answers:
  "If we apply intervention X to this patient RIGHT NOW,
   what happens to their risk in the next 6–24 hours?"

Workflow:
  1. Load current patient vitals sequence
  2. Apply intervention delta to relevant feature(s)
  3. Re-score with LSTM risk model (new risk curve)
  4. Adjust with CATE from causal model (patient-specific effect)
  5. Return: baseline_risk, cf_risk, delta, confidence_interval

Usage (standalone):
    from simulator.counterfactual import CounterfactualSimulator
    sim = CounterfactualSimulator()
    result = sim.simulate(patient_X, intervention="oxygen_increase", magnitude=10)
"""

import json
import numpy as np
import torch
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.temporal.risk_model import ORIONRiskModel

# ── Feature column registry ────────────────────────────────────────────────
FEATURE_COLS = [
    "heart_rate", "sbp", "dbp", "spo2", "temp_f", "resp_rate",
    "lactate", "wbc", "creatinine", "bicarbonate", "glucose",
]
FEATURE_IDX = {col: i for i, col in enumerate(FEATURE_COLS)}

# ── Intervention definitions ───────────────────────────────────────────────
INTERVENTIONS = {
    "oxygen_increase": {
        "feature":     "spo2",
        "description": "Increase oxygen / FiO₂",
        "unit":        "%",
        "default":     5.0,
        "min":         1.0,
        "max":         20.0,
        "clip":        (50.0, 100.0),
        "direction":   +1,   # positive = beneficial
    },
    "fluid_bolus": {
        "feature":     "sbp",
        "description": "IV Fluid Bolus (500 mL NS)",
        "unit":        "mmHg SBP change",
        "default":     5.0,
        "min":         1.0,
        "max":         20.0,
        "clip":        (40.0, 280.0),
        "direction":   +1,
    },
    "vasopressor": {
        "feature":     "sbp",
        "description": "Vasopressor administration",
        "unit":        "mmHg SBP change",
        "default":     10.0,
        "min":         5.0,
        "max":         30.0,
        "clip":        (40.0, 280.0),
        "direction":   +1,
    },
}


@dataclass
class SimulationResult:
    intervention:      str
    magnitude:         float
    baseline_risks:    list[float]          # per-step risk curve [T]
    cf_risks:          list[float]          # counterfactual risk curve [T]
    baseline_final:    float                # final aggregated baseline risk
    cf_final:          float                # final aggregated CF risk
    risk_delta:        float                # cf_final - baseline_final
    cate_adjustment:   float                # causal forest adjustment
    ci_lower:          float                # 95% CI lower bound
    ci_upper:          float                # 95% CI upper bound
    recommendation:    str = ""
    confidence_label:  str = ""


class CounterfactualSimulator:
    """
    Loads trained LSTM + CATE models and performs counterfactual simulation.
    """

    def __init__(
        self,
        model_path:  str = "models/temporal/best_model.pt",
        causal_dir:  str = "models/causal",
        device:      str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._load_risk_model(model_path)
        self._load_cate_models(causal_dir)
        self.scaler = self._try_load_scaler()

    # ── Loaders ─────────────────────────────────────────────────────────────

    def _load_risk_model(self, path: str):
        path = Path(path)
        if not path.exists():
            print(f"[WARN] Risk model not found at {path}. Using random weights.")
            self.risk_model = ORIONRiskModel(n_features=len(FEATURE_COLS)).to(self.device)
            self.risk_model.eval()
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        meta = ckpt.get("meta", {})
        n_features = meta.get("n_features", len(FEATURE_COLS))
        self.risk_model = ORIONRiskModel(n_features=n_features).to(self.device)
        self.risk_model.load_state_dict(ckpt["model_state"])
        self.risk_model.eval()
        print(f"✓ Risk model loaded (val AUC={ckpt.get('val_auc', '?'):.4f})")

    def _load_cate_models(self, causal_dir: str):
        self.cate_models = {}
        for name in ["oxygen_increase", "fluid_bolus"]:
            p = Path(causal_dir) / f"cate_{name}.pkl"
            if p.exists():
                self.cate_models[name] = joblib.load(p)
                print(f"✓ CATE model loaded: {name}")

    def _try_load_scaler(self):
        p = Path("data/processed/tensors/scaler.pkl")
        if p.exists():
            return joblib.load(p)
        return None

    # ── Core simulation ──────────────────────────────────────────────────────

    def _score(self, X_np: np.ndarray):
        """Score a [T, F] numpy array. Returns (step_risks, final_risk)."""
        if self.scaler is not None:
            T, F = X_np.shape
            X_np = self.scaler.transform(X_np.reshape(-1, F)).reshape(T, F)
        tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            step_risks, final_risk = self.risk_model(tensor)
        return (
            step_risks.squeeze(0).cpu().numpy().tolist(),
            float(final_risk.squeeze(0).cpu().numpy()),
        )

    def simulate(
        self,
        patient_X:       np.ndarray,           # [T, F] raw vitals (pre-scaling)
        intervention:    str,
        magnitude:       float | None = None,
        patient_context: dict | None = None,   # age, comorbidity etc for CATE
    ) -> SimulationResult:
        """
        Run a counterfactual simulation.

        Args:
            patient_X:       Last 24h vitals, shape [T, n_features]
            intervention:    One of INTERVENTIONS keys
            magnitude:       How much to change the feature (unit = INTERVENTIONS[x]['unit'])
            patient_context: Dict with keys age, comorbidity_score, baseline_hr,
                             baseline_sbp, baseline_spo2  (used for CATE)
        Returns:
            SimulationResult dataclass
        """
        if intervention not in INTERVENTIONS:
            raise ValueError(f"Unknown intervention '{intervention}'. "
                             f"Choose from: {list(INTERVENTIONS.keys())}")

        cfg       = INTERVENTIONS[intervention]
        magnitude = magnitude if magnitude is not None else cfg["default"]
        feat_idx  = FEATURE_IDX.get(cfg["feature"])

        # ── Baseline score ──
        baseline_steps, baseline_final = self._score(patient_X.copy())

        # ── Apply intervention to last N hours (last 6h → simulate sudden change) ──
        cf_X = patient_X.copy()
        if feat_idx is not None:
            lo, hi = cfg["clip"]
            cf_X[-6:, feat_idx] = np.clip(
                cf_X[-6:, feat_idx] + magnitude * cfg["direction"],
                lo, hi
            )

        cf_steps, cf_final = self._score(cf_X)

        # ── CATE adjustment ─────────────────────────────────────────────────
        cate_adj = 0.0
        if intervention in self.cate_models and patient_context:
            CONFOUNDER_KEYS = [
                "age", "comorbidity_score",
                "baseline_hr", "baseline_sbp", "baseline_spo2"
            ]
            ctx_vec = np.array(
                [[patient_context.get(k, 0) for k in CONFOUNDER_KEYS]],
                dtype=np.float32
            )
            cate_adj = float(self.cate_models[intervention].effect(ctx_vec)[0])

        # ── Uncertainty (bootstrap interval approximation) ──────────────────
        risk_delta = (cf_final + cate_adj) - baseline_final
        noise_std  = max(abs(risk_delta) * 0.25, 0.02)   # ±25% or ±2pp floor
        ci_lower   = risk_delta - 1.96 * noise_std
        ci_upper   = risk_delta + 1.96 * noise_std

        # ── Recommendation label ─────────────────────────────────────────────
        if risk_delta < -0.05:
            rec   = f"✅ Recommended — reduces risk by {abs(risk_delta)*100:.1f}pp"
            conf  = "High benefit"
        elif risk_delta < 0:
            rec   = f"⚠️ Marginal benefit ({abs(risk_delta)*100:.1f}pp reduction)"
            conf  = "Modest benefit"
        else:
            rec   = f"❌ Not beneficial (risk increases {risk_delta*100:.1f}pp)"
            conf  = "No benefit"

        return SimulationResult(
            intervention   = intervention,
            magnitude      = magnitude,
            baseline_risks = baseline_steps,
            cf_risks       = cf_steps,
            baseline_final = baseline_final,
            cf_final       = cf_final + cate_adj,
            risk_delta     = risk_delta,
            cate_adjustment= cate_adj,
            ci_lower       = ci_lower,
            ci_upper       = ci_upper,
            recommendation = rec,
            confidence_label = conf,
        )

    def rank_interventions(
        self,
        patient_X: np.ndarray,
        patient_context: dict | None = None,
    ) -> list[SimulationResult]:
        """
        Simulate all available interventions and rank by risk reduction.
        """
        results = []
        for name in INTERVENTIONS:
            try:
                res = self.simulate(patient_X, name,
                                    patient_context=patient_context)
                results.append(res)
            except Exception as e:
                print(f"[WARN] Skipping {name}: {e}")
        results.sort(key=lambda r: r.risk_delta)
        return results


if __name__ == "__main__":
    # Smoke test with random patient
    sim = CounterfactualSimulator()
    X   = np.random.randn(24, len(FEATURE_COLS)).astype(np.float32)
    ctx = {"age": 65, "comorbidity_score": 3,
           "baseline_hr": 92, "baseline_sbp": 88, "baseline_spo2": 91}

    print("\n── Oxygen Increase Simulation ──")
    result = sim.simulate(X, "oxygen_increase", magnitude=10, patient_context=ctx)
    print(f"  Baseline risk : {result.baseline_final:.4f}")
    print(f"  CF risk       : {result.cf_final:.4f}")
    print(f"  Delta         : {result.risk_delta:+.4f}")
    print(f"  CI            : [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"  {result.recommendation}")

    print("\n── All Intervention Ranking ──")
    ranked = sim.rank_interventions(X, ctx)
    for r in ranked:
        print(f"  {r.intervention:<20} Δ={r.risk_delta:+.4f}  {r.confidence_label}")
