"""
ORION — Streamlit Decision Interface (Phase 3)
===============================================
Run:
    streamlit run ui/app.py

Features:
  - Patient selector (demo stays or upload)
  - Live 24h vitals timeline
  - Real-time risk curve
  - Intervention sliders → counterfactual simulation
  - Before vs After risk comparison with confidence bands
  - Best Action ranking panel
  - Disclaimer footer
"""

import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.counterfactual import (
    CounterfactualSimulator, INTERVENTIONS, FEATURE_COLS, SimulationResult
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ORION — ICU Intervention Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0f172a; }

  section[data-testid="stSidebar"] {
    background: #1e293b;
    border-right: 1px solid #334155;
  }

  .metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; }
  .metric-label { font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase; }

  .risk-delta-pos { color: #10b981; }
  .risk-delta-neg { color: #ef4444; }

  .section-header {
    font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
    border-left: 3px solid #7c3aed; padding-left: 0.6rem;
    margin: 1.2rem 0 0.6rem;
  }

  .disclaimer {
    background: #1e293b; border: 1px solid #ef4444;
    border-radius: 8px; padding: 0.75rem 1rem;
    color: #ef4444; font-size: 0.8rem; margin-top: 2rem;
  }

  .badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
  }
  .badge-purple  { background: #4c1d95; color: #c4b5fd; }
  .badge-green   { background: #064e3b; color: #6ee7b7; }
  .badge-red     { background: #7f1d1d; color: #fca5a5; }
  .badge-yellow  { background: #78350f; color: #fcd34d; }

  div[data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ORION models…")
def load_simulator():
    model_path = ROOT / "models" / "temporal" / "best_model.pt"
    causal_dir = ROOT / "models" / "causal"
    return CounterfactualSimulator(
        model_path=str(model_path),
        causal_dir=str(causal_dir),
    )


def load_demo_patients() -> dict[str, np.ndarray]:
    """Load processed stay time-series from data/processed/timeseries."""
    ts_dir = ROOT / "data" / "processed" / "timeseries"
    patients = {}
    if ts_dir.exists():
        files = sorted(ts_dir.glob("*.parquet"))[:20]  # first 20 for demo
        for f in files:
            try:
                df = pd.read_parquet(f)
                arr = np.zeros((24, len(FEATURE_COLS)), dtype=np.float32)
                for i, col in enumerate(FEATURE_COLS):
                    if col in df.columns:
                        vals = df[col].dropna().values
                        if len(vals) >= 24:
                            arr[:, i] = vals[-24:]
                        elif len(vals) > 0:
                            arr[-len(vals):, i] = vals
                patients[f"Stay {f.stem}"] = arr
            except Exception:
                continue

    # Always include synthetic demo patients
    rng = np.random.default_rng(42)
    DEMO = {
        "Demo: Sepsis (High Risk)": _synthetic_patient(
            rng, hr_base=115, sbp_base=82, spo2_base=88, risk_level="high"),
        "Demo: Post-Op (Medium Risk)": _synthetic_patient(
            rng, hr_base=98, sbp_base=105, spo2_base=93, risk_level="medium"),
        "Demo: Recovering (Low Risk)": _synthetic_patient(
            rng, hr_base=78, sbp_base=122, spo2_base=97, risk_level="low"),
    }
    patients = {**DEMO, **patients}
    return patients


def _synthetic_patient(rng, hr_base, sbp_base, spo2_base, risk_level) -> np.ndarray:
    T, F = 24, len(FEATURE_COLS)
    X = np.zeros((T, F), dtype=np.float32)
    noise = rng.normal(0, 1, (T,))
    # heart_rate, sbp, dbp, spo2, temp_f, resp_rate, lactate, wbc, creatinine, bicarbonate, glucose
    trends = {"high": np.linspace(0, 8, T), "medium": np.zeros(T), "low": np.linspace(0, -3, T)}
    t = trends[risk_level]
    X[:, 0] = hr_base  + t      + rng.normal(0, 3,  T)   # HR
    X[:, 1] = sbp_base - t*0.5  + rng.normal(0, 5,  T)   # SBP
    X[:, 2] = (sbp_base * 0.6)  + rng.normal(0, 3,  T)   # DBP
    X[:, 3] = spo2_base - t*0.3 + rng.normal(0, 1,  T)   # SpO2
    X[:, 4] = 98.6               + rng.normal(0, 0.5,T)   # Temp
    X[:, 5] = 18 + t*0.3         + rng.normal(0, 2,  T)   # RR
    X[:, 6] = (1.5 if risk_level=="high" else 0.8) + rng.normal(0, 0.2, T)   # Lactate
    X[:, 7] = 12                  + rng.normal(0, 2,  T)   # WBC
    X[:, 8] = 1.2                 + rng.normal(0, 0.2,T)   # Creatinine
    X[:, 9] = 22                  + rng.normal(0, 2,  T)   # Bicarb
    X[:,10] = 120                 + rng.normal(0, 10, T)   # Glucose
    return np.clip(X, 0, None)


def risk_color(risk: float) -> str:
    if risk < 0.3:  return "#10b981"
    if risk < 0.6:  return "#f59e0b"
    return "#ef4444"


def delta_badge(delta: float) -> str:
    pct = delta * 100
    if delta < -0.05:
        return f'<span class="badge badge-green">▼ {abs(pct):.1f}pp</span>'
    elif delta < 0:
        return f'<span class="badge badge-yellow">▼ {abs(pct):.1f}pp</span>'
    else:
        return f'<span class="badge badge-red">▲ {pct:.1f}pp</span>'


# ── Plots ─────────────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    """Convert '#rrggbb' → 'rgba(r,g,b,alpha)' for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_vitals_timeline(X: np.ndarray) -> go.Figure:
    hours = list(range(len(X)))
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True,
                        vertical_spacing=0.08, horizontal_spacing=0.1,
                        subplot_titles=["Heart Rate (bpm)", "Blood Pressure (mmHg)",
                                        "SpO₂ (%)", "Respiratory Rate",
                                        "Lactate (mmol/L)", "Glucose (mg/dL)"])
    DARK_LAYOUT = dict(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
                       font=dict(color="#e2e8f0", family="Inter"),
                       margin=dict(l=30, r=10, t=40, b=20))

    def add_line(row, col, feat_idx, color, name, y_data=None):
        y = y_data if y_data is not None else X[:, feat_idx]
        fig.add_trace(go.Scatter(
            x=hours, y=y, name=name,
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=hex_to_rgba(color, 0.10),
            mode="lines",
        ), row=row, col=col)

    add_line(1, 1, 0,  "#ef4444", "HR")
    # SBP + DBP as range
    fig.add_trace(go.Scatter(x=hours+hours[::-1],
                             y=list(X[:,1])+list(X[:,2])[::-1],
                             fill="toself", fillcolor="rgba(14,165,233,0.15)",
                             line=dict(color="#0ea5e9", width=0),
                             name="BP range"), row=1, col=2)
    fig.add_trace(go.Scatter(x=hours, y=X[:,1], line=dict(color="#0ea5e9", width=2),
                             name="SBP"), row=1, col=2)
    add_line(2, 1, 3,  "#10b981", "SpO₂")
    add_line(2, 2, 5,  "#f59e0b", "RR")
    add_line(3, 1, 6,  "#a855f7", "Lactate")
    add_line(3, 2, 10, "#06b6d4", "Glucose")

    fig.update_layout(**DARK_LAYOUT, height=380, showlegend=False,
                      title_text="Patient Vitals — Last 24 Hours",
                      title_font_size=13)
    for ax in fig.layout:
        if ax.startswith("xaxis"):
            fig.layout[ax].update(gridcolor="#334155", showgrid=True)
        if ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor="#334155", showgrid=True)
    return fig


def plot_risk_curve(baseline_steps: list, cf_steps: list | None = None,
                    intervention_label: str = "") -> go.Figure:
    hours = list(range(len(baseline_steps)))
    DARK_LAYOUT = dict(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
                       font=dict(color="#e2e8f0", family="Inter"),
                       margin=dict(l=30, r=20, t=45, b=30))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=baseline_steps, name="Baseline Risk",
                             line=dict(color="#64748b", width=2, dash="dot"),
                             fill="tozeroy", fillcolor="rgba(100,116,139,0.1)"))

    if cf_steps:
        fig.add_trace(go.Scatter(x=hours, y=cf_steps,
                                 name=f"After: {intervention_label}",
                                 line=dict(color="#7c3aed", width=2.5),
                                 fill="tozeroy", fillcolor="rgba(124,58,237,0.15)"))

    # Risk bands
    for y_val, label, color in [(0.3, "Low/Med", "#10b981"),
                                 (0.6, "Med/High", "#f59e0b")]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="right")

    fig.update_layout(**DARK_LAYOUT, height=280,
                      title_text="Risk Score Trajectory",
                      title_font_size=13,
                      yaxis=dict(range=[0, 1], gridcolor="#334155",
                                 title="Risk Score"),
                      xaxis=dict(gridcolor="#334155", title="Hour"),
                      legend=dict(bgcolor="rgba(0,0,0,0)", x=0, y=1))
    return fig


def plot_before_after(result: SimulationResult) -> go.Figure:
    labels  = ["Baseline Risk", f"After {result.intervention.replace('_', ' ').title()}"]
    values  = [result.baseline_final, result.cf_final]
    colors  = [risk_color(result.baseline_final), risk_color(result.cf_final)]
    ci_low  = [0, abs(result.ci_lower)]
    ci_high = [0, abs(result.ci_upper)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        error_y=dict(type="data", symmetric=False,
                     array=ci_high, arrayminus=ci_low, color="#94a3b8"),
        text=[f"{v:.3f}" for v in values],
        textposition="outside", textfont=dict(color="white", size=14),
        width=0.5,
    ))

    fig.update_layout(
        plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", family="Inter"),
        margin=dict(l=20, r=20, t=45, b=20),
        height=240,
        title_text="Risk Before vs After Intervention",
        title_font_size=13,
        yaxis=dict(range=[0, 1.1], gridcolor="#334155", title="Risk Score"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def plot_intervention_ranking(results: list[SimulationResult]) -> go.Figure:
    names  = [r.intervention.replace("_", " ").title() for r in results]
    deltas = [r.risk_delta * 100 for r in results]
    colors = [PALETTE_RANK(d) for d in deltas]

    fig = go.Figure(go.Bar(
        x=deltas, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{d:+.1f}pp" for d in deltas],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.add_vline(x=0, line_color="#94a3b8", line_dash="dot")
    fig.update_layout(
        plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", family="Inter"),
        height=200,
        margin=dict(l=20, r=60, t=30, b=20),
        title_text="Intervention Ranking (risk reduction)",
        title_font_size=13,
        xaxis=dict(gridcolor="#334155", title="Risk Change (percentage points)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def PALETTE_RANK(delta: float) -> str:
    if delta < -5:   return "#10b981"
    if delta < 0:    return "#84cc16"
    if delta < 5:    return "#f59e0b"
    return "#ef4444"


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
        <span style="font-size:2.2rem;">🔬</span>
        <div>
            <h1 style="margin:0; font-size:1.8rem; font-weight:700;
                       background:linear-gradient(135deg,#7c3aed,#0ea5e9);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                ORION
            </h1>
            <p style="margin:0; color:#94a3b8; font-size:0.85rem;">
                Causal AI for ICU Intervention Planning
                &nbsp;<span class="badge badge-purple">Research Prototype</span>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Load resources ──
    sim      = load_simulator()
    patients = load_demo_patients()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏥 Patient Selection")
        patient_name = st.selectbox("Select patient", options=list(patients.keys()))
        patient_X    = patients[patient_name]

        st.markdown("---")
        st.markdown("### 💊 Intervention Controls")

        intervention = st.selectbox(
            "Intervention type",
            options=list(INTERVENTIONS.keys()),
            format_func=lambda k: INTERVENTIONS[k]["description"],
        )
        cfg = INTERVENTIONS[intervention]
        magnitude = st.slider(
            f"Magnitude ({cfg['unit']})",
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=0.5,
        )

        st.markdown("---")
        st.markdown("### 👤 Patient Context (CATE)")
        age              = st.slider("Age",               30, 90, 65)
        comorbidity      = st.slider("Comorbidity Score", 0, 8, 3)
        baseline_hr_val  = st.slider("Baseline HR (bpm)", 40, 180, int(patient_X[-1, 0]) or 90)
        baseline_sbp_val = st.slider("Baseline SBP (mmHg)", 40, 200, int(patient_X[-1, 1]) or 110)
        baseline_spo2_val= st.slider("Baseline SpO₂ (%)",  70, 100, int(patient_X[-1, 3]) or 94)

        simulate_btn = st.button("⚡ Run Simulation", use_container_width=True,
                                 type="primary")
        rank_btn     = st.button("📊 Rank All Interventions", use_container_width=True)

    # ── State ────────────────────────────────────────────────────────────────
    patient_context = {
        "age": age, "comorbidity_score": comorbidity,
        "baseline_hr": baseline_hr_val,
        "baseline_sbp": baseline_sbp_val,
        "baseline_spo2": baseline_spo2_val,
    }

    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "ranked" not in st.session_state:
        st.session_state["ranked"] = None

    # ── Run Simulation ────────────────────────────────────────────────────────
    if simulate_btn:
        with st.spinner("Running counterfactual simulation …"):
            result = sim.simulate(patient_X, intervention,
                                  magnitude=magnitude,
                                  patient_context=patient_context)
        st.session_state["result"] = result
        st.session_state["ranked"] = None

    if rank_btn:
        with st.spinner("Ranking all interventions …"):
            ranked = sim.rank_interventions(patient_X, patient_context)
        st.session_state["ranked"] = ranked
        st.session_state["result"] = None

    # ── Main Layout ───────────────────────────────────────────────────────────
    result = st.session_state["result"]
    ranked = st.session_state["ranked"]

    # Vitals timeline — always shown
    st.markdown('<div class="section-header">📊 Patient Vitals Timeline</div>',
                unsafe_allow_html=True)
    st.plotly_chart(plot_vitals_timeline(patient_X), use_container_width=True)

    # Risk curve
    st.markdown('<div class="section-header">📈 Risk Score Trajectory</div>',
                unsafe_allow_html=True)
    if result:
        st.plotly_chart(
            plot_risk_curve(result.baseline_risks, result.cf_risks,
                            INTERVENTIONS[intervention]["description"]),
            use_container_width=True
        )
    else:
        # Show baseline only — _score() handles no_grad internally
        _, baseline_final = sim._score(patient_X)
        st.plotly_chart(plot_risk_curve(
            [float(baseline_final)] * 24
        ), use_container_width=True)

    # ── Results panel ─────────────────────────────────────────────────────────
    if result:
        st.markdown('<div class="section-header">🔬 Simulation Results</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{risk_color(result.baseline_final)}">
                    {result.baseline_final:.3f}
                </div>
                <div class="metric-label">Baseline Risk</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{risk_color(result.cf_final)}">
                    {result.cf_final:.3f}
                </div>
                <div class="metric-label">Post-Intervention Risk</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            d = result.risk_delta
            color = "#10b981" if d < 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">
                    {d*100:+.1f}pp
                </div>
                <div class="metric-label">Risk Delta</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.2rem; color:#94a3b8">
                    [{result.ci_lower*100:+.1f}, {result.ci_upper*100:+.1f}]
                </div>
                <div class="metric-label">95% CI (pp)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"<br><b>Recommendation:</b> {result.recommendation}",
                    unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(plot_before_after(result), use_container_width=True)
        with col_b:
            st.markdown("**CATE Adjustment**")
            st.info(
                f"Patient-specific causal effect (CATE): **{result.cate_adjustment:+.4f}**\n\n"
                f"This adjusts the model's prediction based on this patient's "
                f"age, comorbidities, and baseline vitals."
            )
            st.markdown("**Confidence**")
            st.success(result.confidence_label)

    # ── Intervention Ranking ──────────────────────────────────────────────────
    if ranked:
        st.markdown('<div class="section-header">🏆 Best Action Now</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_intervention_ranking(ranked), use_container_width=True)

        st.markdown("**Detailed Ranking**")
        rows = []
        for i, r in enumerate(ranked):
            badge = delta_badge(r.risk_delta)
            rows.append({
                "Rank": i + 1,
                "Intervention": INTERVENTIONS[r.intervention]["description"],
                "Risk Delta": badge,
                "Baseline → After": f"{r.baseline_final:.3f} → {r.cf_final:.3f}",
                "95% CI": f"[{r.ci_lower*100:+.1f}, {r.ci_upper*100:+.1f}] pp",
                "Verdict": r.recommendation.split("—")[0].strip(),
            })
        df = pd.DataFrame(rows)
        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Research Prototype — Not for Clinical Use.</strong>
        ORION is an academic decision-support research system.
        Outputs are not validated for clinical decision-making.
        All intervention recommendations must be reviewed by qualified clinicians.
        Data source: MIMIC-IV (PhysioNet).
    </div>
    """, unsafe_allow_html=True)




if __name__ == "__main__":
    main()
