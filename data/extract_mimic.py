"""
ORION — Phase 1: MIMIC-IV Data Extraction
==========================================
Loads the MIMIC-IV Demo dataset (no credential wait) and extracts:
  - Hourly vitals per ICU stay (HR, BP, SpO2, Temp, RR)
  - Key lab values (lactate, WBC, creatinine)
  - Fluid/oxygen interventions
  - Outcome labels (in-hospital mortality + 24h sepsis proxy)

Usage:
    python data/extract_mimic.py --mimic_dir data/raw/mimic-iv-demo --out_dir data/processed

Download MIMIC-IV Demo (no credentialing):
    wget -r -N -c -np https://physionet.org/files/mimic-iv-demo/2.2/
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Vital sign item IDs (chartevents) ──────────────────────────────────────
VITAL_ITEMIDS = {
    220045: "heart_rate",
    220179: "sbp",           # systolic BP
    220180: "dbp",           # diastolic BP
    220277: "spo2",
    223761: "temp_f",
    220210: "resp_rate",
}

# ── Lab item IDs (labevents) ────────────────────────────────────────────────
LAB_ITEMIDS = {
    50813: "lactate",
    51301: "wbc",
    50912: "creatinine",
    50882: "bicarbonate",
    50931: "glucose",
}

# ── Oxygen / Fluid item IDs ─────────────────────────────────────────────────
O2_ITEMIDS   = {223835, 229314, 226253}   # FiO2 / O2 flow / O2 device
FLUID_ITEMIDS= {225158, 220949, 225828}   # NS, D5W, LR bolus


def load_csv(mimic_dir: Path, table: str, usecols=None) -> pd.DataFrame:
    """Load a MIMIC-IV table (hosp or icu subfolder)."""
    for sub in ["icu", "hosp"]:
        p = mimic_dir / sub / f"{table}.csv.gz"
        if not p.exists():
            p = mimic_dir / sub / f"{table}.csv"
        if p.exists():
            return pd.read_csv(p, usecols=usecols, low_memory=False)
    raise FileNotFoundError(f"Table '{table}' not found in {mimic_dir}")


def extract_vitals(mimic_dir: Path) -> pd.DataFrame:
    print("→ Loading chartevents …")
    cols = ["subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"]
    ce = load_csv(mimic_dir, "chartevents", usecols=cols)
    ce = ce[ce["itemid"].isin(VITAL_ITEMIDS)].dropna(subset=["valuenum"])
    ce["feature"] = ce["itemid"].map(VITAL_ITEMIDS)
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    return ce[["stay_id", "charttime", "feature", "valuenum"]]


def extract_labs(mimic_dir: Path) -> pd.DataFrame:
    print("→ Loading labevents …")
    cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    le = load_csv(mimic_dir, "labevents", usecols=cols)
    le = le[le["itemid"].isin(LAB_ITEMIDS)].dropna(subset=["valuenum"])
    le["feature"] = le["itemid"].map(LAB_ITEMIDS)
    le["charttime"] = pd.to_datetime(le["charttime"])
    # Join stay_id via icustays
    icu = load_csv(mimic_dir, "icustays",
                   usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    icu["intime"]  = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])
    le = le.merge(icu[["hadm_id", "stay_id", "intime", "outtime"]],
                  on="hadm_id", how="inner")
    le = le[(le["charttime"] >= le["intime"]) & (le["charttime"] <= le["outtime"])]
    return le[["stay_id", "charttime", "feature", "valuenum"]]


def extract_interventions(mimic_dir: Path) -> pd.DataFrame:
    print("→ Loading inputevents (fluids / O2) …")
    cols = ["stay_id", "starttime", "itemid", "amount", "amountuom"]
    ie = load_csv(mimic_dir, "inputevents", usecols=cols)
    ie["starttime"] = pd.to_datetime(ie["starttime"])
    ie["is_fluid"] = ie["itemid"].isin(FLUID_ITEMIDS).astype(int)
    ie["is_o2"]    = ie["itemid"].isin(O2_ITEMIDS).astype(int)
    ie = ie[(ie["is_fluid"] == 1) | (ie["is_o2"] == 1)]
    return ie[["stay_id", "starttime", "is_fluid", "is_o2", "amount"]]


def extract_outcomes(mimic_dir: Path) -> pd.DataFrame:
    print("→ Loading outcomes …")
    adm = load_csv(mimic_dir, "admissions",
                   usecols=["hadm_id", "hospital_expire_flag", "deathtime"])
    icu = load_csv(mimic_dir, "icustays",
                   usecols=["hadm_id", "stay_id", "los"])
    df  = icu.merge(adm, on="hadm_id", how="left")
    df["mortality"] = df["hospital_expire_flag"].fillna(0).astype(int)
    # Sepsis proxy: LOS > 3 days AND died
    df["sepsis_proxy"] = ((df["los"] > 3) & (df["mortality"] == 1)).astype(int)
    return df[["stay_id", "hadm_id", "los", "mortality", "sepsis_proxy"]]


def build_hourly_timeseries(vitals: pd.DataFrame, labs: pd.DataFrame,
                             icu_stays: pd.DataFrame, out_dir: Path):
    """
    For each ICU stay, build a DataFrame with hourly rows × feature columns.
    Saves one Parquet file per stay.
    """
    all_events = pd.concat([vitals, labs], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stay_ids = all_events["stay_id"].unique()
    print(f"→ Building hourly time-series for {len(stay_ids)} stays …")

    saved = 0
    for sid in tqdm(stay_ids):
        stay_row = icu_stays[icu_stays["stay_id"] == sid]
        if stay_row.empty:
            continue
        intime  = pd.to_datetime(stay_row["intime"].values[0])
        outtime = pd.to_datetime(stay_row["outtime"].values[0])

        df = all_events[all_events["stay_id"] == sid].copy()
        df["hour"] = ((df["charttime"] - intime).dt.total_seconds() / 3600).astype(int)
        df = df[(df["hour"] >= 0) & (df["hour"] <= 168)]   # cap at 7 days

        # Pivot: hour × feature (mean if multiple readings per hour)
        pivot = df.pivot_table(index="hour", columns="feature",
                               values="valuenum", aggfunc="mean")
        pivot = pivot.reindex(range(int(df["hour"].max()) + 1))

        if len(pivot) < 6:   # skip stays with <6 hours of data
            continue

        pivot.index.name = "hour_offset"
        pivot["stay_id"] = sid
        pivot.to_parquet(out_dir / f"{sid}.parquet")
        saved += 1

    print(f"✓ Saved {saved} stay time-series to {out_dir}")


def main(mimic_dir: str, out_dir: str):
    mimic_dir = Path(mimic_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vitals  = extract_vitals(mimic_dir)
    labs    = extract_labs(mimic_dir)
    intrvns = extract_interventions(mimic_dir)
    outcomes= extract_outcomes(mimic_dir)

    icu = load_csv(mimic_dir, "icustays",
                   usecols=["stay_id", "hadm_id", "subject_id", "intime", "outtime"])
    icu["intime"]  = pd.to_datetime(icu["intime"])
    icu["outtime"] = pd.to_datetime(icu["outtime"])

    # Save flat files
    outcomes.to_parquet(out_dir / "outcomes.parquet", index=False)
    intrvns.to_parquet(out_dir  / "interventions.parquet", index=False)
    print(f"✓ outcomes.parquet: {len(outcomes)} stays")
    print(f"✓ interventions.parquet: {len(intrvns)} events")

    # Build per-stay hourly files
    ts_dir = out_dir / "timeseries"
    build_hourly_timeseries(vitals, labs, icu, ts_dir)
    print("\n✅ Extraction complete. Next: python data/preprocess.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_dir", default="data/raw/mimic-iv-demo")
    parser.add_argument("--out_dir",   default="data/processed")
    args = parser.parse_args()
    main(args.mimic_dir, args.out_dir)
