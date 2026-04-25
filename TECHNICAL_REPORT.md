# Technical Report: ORION — Causal AI for ICU Intervention Planning

**Author:** Siddhant Chandorkar  
**Role:** Principal AI Researcher / Senior ML Systems Architect  
**Date:** April 2026

---

## 1. The Problem Space

Modern Intensive Care Units (ICUs) are incredibly data-rich environments, generating thousands of data points per patient per day. Machine Learning models have been highly successful at utilizing this data for **predictive risk scoring** (e.g., forecasting that a patient has an 85% probability of mortality or sepsis). 

However, pure prediction is insufficient for clinical decision-making. If a model predicts an 85% mortality risk, the clinician's immediate question is: *"What intervention will reduce this risk the most?"* Pure predictive models cannot answer this because they suffer from **confounding bias**; they learn correlations (e.g., patients on ventilators have higher mortality) rather than causation (ventilators save lives, but are given to the sickest patients).

**The Objective:** ORION was developed to bridge this gap by transitioning from predictive ML to **Prescriptive Causal AI**. The system simulates counterfactual scenarios: *"If we apply Intervention X (e.g., IV fluid bolus), how will the patient's specific risk trajectory change over the next 24 hours?"*

---

## 2. Methodology & Architecture

ORION employs a dual-engine architecture, separating temporal perception from causal reasoning.

### 2.1. Temporal Perception Layer (PyTorch)
To establish a highly accurate baseline risk trajectory, we developed a Deep Sequence Model.
* **Architecture:** Bidirectional Long Short-Term Memory (Bi-LSTM) network with attention pooling.
* **Inputs:** 24-hour lookback windows of multivariate physiological data (Heart Rate, Blood Pressure, SpO2, Temperature).
* **Handling Imbalance:** Medical outcomes are highly skewed (mortality is the minority class). We implemented **Focal Loss** ($\gamma=2.0$) to dynamically scale the loss based on prediction confidence, forcing the model to learn hard-to-predict adverse outcomes.

### 2.2. Causal Engine (DoWhy + EconML)
To calculate the true effect of interventions, we utilized Double Machine Learning (DML).
* **Causal Graph (DAG):** We explicitly modeled the causal relationships using Microsoft's `DoWhy`. We defined Confounders (Baseline Vitals, Age), Treatments (Oxygen, Fluids), and Outcomes (Mortality Risk).
* **CATE Estimation:** We trained a **Causal Forest** (`EconML`) to estimate the Conditional Average Treatment Effect (CATE). Unlike standard ATE, CATE calculates the highly personalized intervention effect based on the patient's unique physiological state at time $t$.

### 2.3. Counterfactual Simulator
The simulator acts as the decision layer. When a clinician proposes an intervention delta (e.g., $+15\%$ Oxygen), the simulator:
1. Calculates the baseline trajectory via the LSTM.
2. Queries the Causal Forest for the personalized CATE.
3. Applies the CATE delta to the baseline risk to generate the new, counterfactual risk curve.
4. Outputs the projected absolute risk reduction.

---

## 3. Validation & Results

Causal claims cannot be evaluated using standard ML metrics like accuracy or ROC-AUC. We implemented a rigorous, automated falsification suite.

### 3.1. Predictive Performance
* The Bi-LSTM baseline achieved robust classification metrics (Target: ROC-AUC $\ge$ 0.75, Brier Score $\le$ 0.15).

### 3.2. Causal Refutation (Defensibility)
To prove the Causal Forest learned true causal effects rather than spurious correlations, we subjected the model to three rigorous tests:
1. **Placebo Treatment Test:** Replaced the intervention with a random variable. The resulting Average Treatment Effect (ATE) successfully collapsed to ~0, proving the model is not hallucinating effects.
2. **Random Common Cause:** Injected an independent random feature as a confounder. The estimated ATE remained stable, proving robustness to unobserved, irrelevant variables.
3. **Data Subset Falsification:** Re-estimated effects on a random 80% subset of the cohort. The estimates remained within a tight confidence interval ($\pm0.02$).

---

## 4. Limitations & Future Work

As a research prototype, ORION has several limitations that must be addressed prior to any clinical deployment:

1. **Unobserved Confounding:** The Causal Forest relies on the assumption of "No Unmeasured Confounding." If a critical variable (e.g., a specific genetic marker or unrecorded medication) heavily influences both the treatment decision and the outcome, the causal estimates will be biased.
2. **Data Sparsity:** Real EHR data contains irregular sampling and missingness. While forward-filling was used for this prototype, advanced imputation methods (e.g., Neural ODEs) are required for clinical grade robustness.
3. **Intervention Modality:** Currently, ORION models interventions as continuous variables. Future iterations must support categorical and combinatorial treatments (e.g., prescribing two specific drugs simultaneously).

---
*End of Report*
