# ORION — Causal AI for Hospital ICU Intervention Planning

> ⚠️ **Research Prototype — Not for Clinical Use.**
> This system is an academic decision-support tool, not validated for real clinical deployment.

---

## 🔬 What is ORION?

ORION answers one question that current ICU systems cannot:

> *"If we give this patient 500ml fluids right now, what happens to their mortality risk in the next 6 hours?"*

This is **counterfactual decision support** — not prediction, but intervention simulation using causal inference.

---

## 🧩 Architecture

```
MIMIC-IV EHR Data
      │
      ▼
┌──────────────────┐      ┌───────────────────┐
│  Temporal Model  │─────▶│   Causal Engine   │
│  (LSTM, 24h)     │      │ (DoWhy + EconML)  │
└──────────────────┘      └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  CF Simulator     │
                          │ (Intervention →   │
                          │  ΔRisk + CI)      │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   Streamlit UI    │
                          │ (Decision Panel)  │
                          └───────────────────┘
```

| Layer | Technology |
|---|---|
| Temporal Risk Model | PyTorch LSTM (bidirectional, attention pooling) |
| Causal Engine | DoWhy (DAG + backdoor), EconML CausalForestDML |
| Counterfactual Simulator | Custom Python engine |
| UI | Streamlit + Plotly |
| Data | MIMIC-IV (PhysioNet) |

---

## ⚡ Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download MIMIC-IV Demo (no credentialing required)
```bash
# Option A: wget
wget -r -N -c -np https://physionet.org/files/mimic-iv-demo/2.2/ -P data/raw/

# Option B: Manual download from physionet.org/content/mimic-iv-demo/2.2/
```

### 3. Extract data
```bash
python data/extract_mimic.py --mimic_dir data/raw/mimic-iv-demo/2.2 --out_dir data/processed
```

### 4. Preprocess
```bash
python data/preprocess.py --ts_dir data/processed/timeseries --out_dir data/processed/tensors
```

### 5. Train risk model
```bash
python models/temporal/train.py --tensor_dir data/processed/tensors --out_dir models/temporal
```

### 6. Run causal engine
```bash
python models/causal/causal_engine.py --processed_dir data/processed --out_dir models/causal
```

### 7. Launch the UI
```bash
streamlit run ui/app.py
```

---

## 📊 Evaluation

### Predictive
```bash
python evaluation/predictive.py
```
Generates: ROC-AUC, PR-AUC, Brier Score, calibration plot.

### Causal Refutation
```bash
python evaluation/causal_refutation.py
```
Generates: ATE comparisons, CATE distributions, placebo + random-common-cause + data-subset refutation scorecard.

---

## 📁 Project Structure

```
orion/
├── data/
│   ├── extract_mimic.py       ← Phase 1: load MIMIC tables
│   ├── preprocess.py          ← Phase 1: normalize + split
│   ├── raw/                   ← MIMIC-IV files (you provide)
│   └── processed/             ← Parquet + PyTorch tensors
├── models/
│   ├── temporal/
│   │   ├── risk_model.py      ← Bidirectional LSTM
│   │   └── train.py           ← Training loop
│   └── causal/
│       └── causal_engine.py   ← DoWhy + EconML
├── simulator/
│   └── counterfactual.py      ← Core CF simulation engine
├── evaluation/
│   ├── predictive.py          ← AUC, calibration, Brier
│   └── causal_refutation.py   ← Refutation scorecard
├── ui/
│   └── app.py                 ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🎯 Evaluation Targets

| Metric | Target |
|---|---|
| ROC-AUC | ≥ 0.75 |
| Brier Score | ≤ 0.15 |
| Placebo refutation | ATE near 0 ✅ |
| Data subset refutation | Stable ±0.02 ✅ |

---

## ⚔️ Key Constraints

- **Cannot claim clinical validity** — frame as "research prototype for decision-support modeling"
- Use temporal train/val/test splits (not random) to prevent data leakage
- All causal estimates must pass at least 2/3 refutation tests before reporting

---

## 🎤 Interview One-Liner

> *"Most systems predict risk. I built a system that simulates how interventions change outcomes using causal inference."*

---

## 📄 License & Attribution

Data: [MIMIC-IV](https://physionet.org/content/mimiciv/) — Johnson, A. et al. PhysioNet.
Research prototype. Not for clinical deployment.
