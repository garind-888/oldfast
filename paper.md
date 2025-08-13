::: {custom-style="MainTitle"}
Title
:::

::: {custom-style="MainSubTitle"}
Subtitle
:::

::: {custom-style="MainPage"}

**Wesley Bennar MD^1^\*, Dorian Garin MD^1^\*, Serban Puricel MD^1^, Pascal Meier MD^1^, Mario Togni MD^1,2^, Stéphane Cook MD^1,2^**

\*Equal contribution

^1^Department of Cardiology, University and Hospital Fribourg, 1708 Fribourg, Switzerland

**Correspondence:**
Stéphane Cook, MD  
Cardiology  
University and Hospital Fribourg  
CH- 1708 Fribourg  
Phone: +41263063809  
Email: stephane.cook@unifr.ch

**Keywords:** 

**Word count:** 

**Abstract word count:** 

**Figures:** 

**Tables:** 

:::

\newpage

# Abstract

\newpage

# Introduction

\newpage

# Methods

## Statistical analysis

Continuous data are presented as mean ± SD or median (IQR) and compared with t-test or Mann-Whitney U; categorical data are n (%) and compared with χ2/Fisher's exact. Survival analysis was performed using Kaplan-Meier curves with log-rank tests for group comparisons. Cox proportional hazards regression was used to calculate hazard ratios (HR) with 95% confidence intervals, with sensitivity analyses using different confounder sets. Proportional hazards assumptions were verified using Schoenfeld residuals. Standardized mortality ratios (SMR) were calculated comparing observed deaths to expected deaths based on Swiss life tables (2020-2023, Human Mortality Database), with 95% CI using exact Poisson methods. Expected survival curves were generated from age-specific mortality rates (qx) using cumulative hazard integration. A two-sided p < 0.05 was considered significant. Analyses were conducted with Python 3.13.0 (NumPy, pandas, statsmodels, scikit-learn, lifelines, SciPy) in Visual Studio Code (version 1.95, © Microsoft 2025) connected to a local PostgreSQL database (version 14.15) administered via DBeaver (version 24.3.4, © DBeaver Corp 2025).

\newpage

# Results

\newpage

# Discussion

\newpage

# Figures

**Figure 1. Study flowchart.**

![Study flowchart](figures/Figure_1.png "Figure 1 – Study flowchart"){width=60%
}

\newpage

# Tables

**Table 1. Baseline Characteristics**

| Variable | Age < 85 (n=1259) | Age ≥85 (n=71) | P-value |
|----------|------------------|----------------|---------|
| **Demographics** | | | |
| Male sex, n (%) | 980 (77.8%) | 32 (45.1%) | <0.001 |
| Age, years, median [IQR] | 61.00 [53.00-70.00] | 87.00 [86.00-89.00] | <0.001 |
| BMI, kg/m², median [IQR] | 26.30 [24.20-29.40] | 24.22 [22.60-26.18] | <0.001 |
| **Cardiovascular Risk Factors** | | | |
| Diabetes mellitus, n (%) | | | 0.698 |
| - Oral treatment | 169 (13.4%) | 10 (14.1%) | |
| - Insulin | 40 (3.2%) | 4 (5.6%) | |
| Dyslipidemia, n (%) | 667 (53.0%) | 25 (35.2%) | 0.005 |
| Hypertension, n (%) | 602 (47.8%) | 47 (66.2%) | 0.004 |
| Smoking status, n (%) | | | <0.001 |
| - Current smoker | 543 (43.1%) | 3 (4.2%) | |
| - Former smoker | 215 (17.1%) | 11 (15.5%) | |
| Family history of CAD, n (%) | 258 (20.5%) | 1 (1.4%) | <0.001 |
| **Medical History** | | | |
| Prior myocardial infarction, n (%) | 128 (10.2%) | 10 (14.1%) | 0.394 |
| Prior CABG, n (%) | 24 (1.9%) | 0 (0.0%) | 0.635 |
| Prior PCI, n (%) | 162 (12.9%) | 13 (18.3%) | 0.254 |
| Heart failure, n (%) | 28 (2.2%) | 6 (8.5%) | 0.004 |
| Prior stroke, n (%) | 22 (1.7%) | 3 (4.2%) | 0.145 |
| Prior bleeding, n (%) | 26 (2.1%) | 7 (9.9%) | <0.001 |
| Cancer, n (%) | 28 (2.2%) | 2 (2.8%) | 0.672 |
| **Culprit Vessel** | | | |
| Circumflex artery, n (%) | 176 (14.0%) | 8 (11.3%) | 0.640 |
| Left anterior descending, n (%) | 555 (44.1%) | 40 (56.3%) | 0.058 |
| Left main, n (%) | 18 (1.4%) | 0 (0.0%) | 0.619 |
| Right coronary artery, n (%) | 499 (39.6%) | 23 (32.4%) | 0.275 |
| **Presentation** | | | |
| Mode of arrival, n (%) | | | 0.016 |
| - Ambulance | 305 (24.2%) | 8 (11.3%) | |
| - Emergency department | 478 (38.0%) | 37 (52.1%) | |
| **Laboratory Values** | | | |
| Peak CK, U/L, median [IQR] | 1425.00 [687.00-2777.00] | 1370.00 [738.00-2521.00] | 0.468 |
| Peak CK-MB, U/L, median [IQR] | 163.00 [80.50-289.50] | 172.50 [102.00-244.00] | 0.627 |
| **Cardiac Function** | | | |
| LVEF, %, median [IQR] | 48.00 [40.00-55.00] | 40.00 [32.00-45.00] | <0.001 |

---

BMI, body mass index; CABG, coronary artery bypass grafting; CAD, coronary artery disease; CK, creatine kinase; CK-MB, creatine kinase-myocardial band; IQR, interquartile range; LVEF, left ventricular ejection fraction; PCI, percutaneous coronary intervention.

*Note:* P-values <0.001 are reported as such for clarity. Continuous variables are presented as median [interquartile range] and categorical variables as n (%).

\newpage

**Table 2. Adjusted Analysis results**

| Outcome | Age < 85 (n=1259) | Age ≥85 (n=71) | Adjusted Difference† | 95% CI | P-value |
|---------|------------------|----------------|---------------------|---------|---------|
| | Mean ± SD | Mean ± SD | | | |
| **Time Intervals (minutes)** | | | | | |
| Pain to balloon | 241.1 ± 167.9 | 304.9 ± 185.5 | 69.2 | [26.0, 112.4] | 0.002 |
| Pain to first medical contact | 141.6 ± 154.0 | 160.3 ± 162.5 | 30.5 | [-11.3, 72.4] | 0.153 |
| First medical contact to balloon | 100.5 ± 54.6 | 138.3 ± 117.5 | 36.7 | [20.6, 52.8] | <0.001 |
| First medical contact to diagnosis | 42.6 ± 46.9 | 53.9 ± 65.5 | 7.9 | [-12.7, 28.4] | 0.451 |
| Diagnosis to balloon | 56.3 ± 21.7 | 65.5 ± 27.3 | 9.1 | [0.8, 17.5] | 0.031 |
| Diagnosis to catheterization lab | 37.4 ± 19.2 | 43.6 ± 19.7 | 5.4 | [-1.7, 12.6] | 0.138 |
| Catheterization lab to balloon | 20.7 ± 12.2 | 21.6 ± 11.1 | 1.2 | [-2.0, 4.4] | 0.456 |


---

CI, confidence interval; CK, creatine kinase; CK-MB, creatine kinase-myocardial band; LVEF, left ventricular ejection fraction; SD, standard deviation.

†Adjusted difference represents age ≥85 minus age <85, adjusted for baseline covariates.

*Note:* All time intervals are presented in minutes. P-values <0.001 are reported as such for clarity.

\newpage

**Table 3. Standardized Mortality Ratios by Age Group**

| Endpoint | Age Group | N | SMR | 95% CI | P-value |
|----------|-----------|---|-----|--------|---------|
| **30-Day All-Cause Mortality** | | | | | |
| | <55 years | 372 | 102.52 | [33.29, 239.25] | <0.001 |
| | 55-64 years | 383 | 89.50 | [47.66, 153.05] | <0.001 |
| | 65-74 years | 320 | 52.77 | [30.74, 84.48] | <0.001 |
| | 75-84 years | 184 | 29.41 | [16.81, 47.76] | <0.001 |
| | ≥85 years | 71 | 10.68 | [4.61, 21.05] | <0.001 |
| **1-Year All-Cause Mortality** | | | | | |
| | <55 years | 361 | 8.96 | [2.91, 20.91] | <0.001 |
| | 55-64 years | 367 | 8.54 | [4.67, 14.32] | <0.001 |
| | 65-74 years | 306 | 6.10 | [3.82, 9.23] | <0.001 |
| | 75-84 years | 180 | 3.57 | [2.23, 5.40] | <0.001 |
| | ≥85 years | 69 | 1.61 | [0.86, 2.76] | 0.134 |
| **5-Year All-Cause Mortality** | | | | | |
| | <55 years | 361 | 6.79 | [3.39, 12.15] | <0.001 |
| | 55-64 years | 367 | 5.33 | [3.34, 8.07] | <0.001 |
| | 65-74 years | 306 | 4.46 | [3.20, 6.05] | <0.001 |
| | 75-84 years | 180 | 2.51 | [1.76, 3.47] | <0.001 |
| | ≥85 years | 69 | 1.43 | [-0.72, 2.85] | 0.121 |

---

CI, confidence interval; SMR, standardized mortality ratio.

*Note:* SMR values represent the ratio of observed to expected deaths based on age- and sex-matched general population data. Values >1 indicate higher mortality than expected, while values <1 indicate lower mortality than expected.

\newpage

**Table 4. Cox Regression Analysis: Mortality Risk by Age Group**

| Endpoint | Age Group | Hazard Ratio† | 95% CI | P-value |
|----------|-----------|---------------|---------|---------|
| **30-Day All-Cause Mortality** | | | | |
| | <55 years | 1.00 | Reference | — |
| | 55-64 years | 2.57 | (0.91-7.24) | 0.075 |
| | 65-74 years | 4.29 | (1.55-11.88) | 0.005 |
| | 75-84 years | 7.39 | (2.60-21.00) | <0.001 |
| | ≥85 years | 10.01 | (3.07-32.71) | <0.001 |
| **1-Year All-Cause Mortality** | | | | |
| | <55 years | 1.00 | Reference | — |
| | 55-64 years | 2.74 | (0.98-7.63) | 0.054 |
| | 65-74 years | 5.47 | (2.03-14.70) | <0.001 |
| | 75-84 years | 9.56 | (3.50-26.07) | <0.001 |
| | ≥85 years | 15.58 | (5.26-46.17) | <0.001 |
| **5-Year All-Cause Mortality** | | | | |
| | <55 years | 1.00 | Reference | — |
| | 55-64 years | 2.03 | (0.98-4.20) | 0.056 |
| | 65-74 years | 4.88 | (2.47-9.64) | <0.001 |
| | 75-84 years | 7.90 | (3.91-15.96) | <0.001 |
| | ≥85 years | 20.03 | (9.41-42.61) | <0.001 |

---
CI, confidence interval.

†Hazard ratios are calculated with the youngest age group (<55 years) as the reference category.

*Note:* All models were adjusted for baseline covariates. P-values <0.001 are reported as such for clarity. The analysis demonstrates a clear age-related gradient in mortality risk across all endpoints, with the oldest patients (≥85 years) having substantially higher risk compared to the youngest group.

\newpage

# Conflict of interest

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

# Author contributions

WB & DG: writing – original draft (lead), analysis (lead). MT, PM, SP: writing – review & editing (equal), data curation (lead), supervision (equal). SC: writing – review & editing (lead), conceptualization (lead), investigation (lead), methodology (lead), supervision (lead).

# Funding

None reported.

# Data availability statement

The raw data supporting the conclusions of this article will be made available by the authors on reasonable request.

\newpage

# References