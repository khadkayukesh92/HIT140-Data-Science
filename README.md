# Bat vs Rat: The Forage Behaviour

CDU - HIT140 Foundations of Data Science (2025)

## Overview

A data analysis project investigating Egyptian Fruit Bats (Rousettus aegyptiacus) foraging behaviour in the presence of Black Rats (Rattus rattus). This repository contains code, analysis notebooks and hypothesis tests used to support Investigation A: whether bats perceive rats as competitors/predators.

## Data

- `dataset1.csv` — One row per bat landing (behavioural annotations).
- `dataset2.csv` — 30-minute observation periods with rat arrivals and aggregated counts.

> Datasets are provided on Learnline (course site). Do **not** upload raw data here if restricted.

## Key analyses / tests

- H1 — Two-proportion z-test: reward success for risk-taking vs risk-avoiding bats
- H2 — Two-sample t-test: approach delay (`bat_landing_to_food`) between groups
- Dataset2 t-tests: bat landings vs food availability; early vs late hours; food availability when rats present vs absent
- Additional diagnostics: histograms, QQ-plots, group sizes, effect sizes, and confidence intervals

## Quick start

```bash
python investigation.py
```

## Requirements

- Python 3.9+
- pandas, numpy, scipy, matplotlib, seaborn

## Results

- Risk-taking bats: much lower success (p1=0.218) vs avoiders (p0=0.843); z ≈ -18.85, 95% CI for p1-p0 entirely negative → rats impose costs.
- Risk-taking bats also show longer approach delays (significant t-test).
- Bat activity declines late at night; presence of rats reduces remaining food in observation periods.


