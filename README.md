# NCAA March Madness 2025: Game Outcome Modelling (Kaggle)

Exploratory analysis and a first modelling approach for Kaggle's **March Machine Learning Mania 2025**: predicting NCAA basketball
game outcomes for both the men's and women's tournaments.

## What's here

| Notebook | What it does |
|---|---|
| [`Initial_EDA.ipynb`](Initial_EDA.ipynb) | Profiles the men's and women's regular-season and tournament results (compact and detailed box scores). Covers score distributions, home/away/neutral effects and team-level averages. |
| [`Ordinals_EDA.ipynb`](Ordinals_EDA.ipynb) | Explores the Massey Ordinals: how each rating system's rank of a team changes over a season, and which systems move most. |
| [`Data Refactor and Initial Model Idea.ipynb`](Data%20Refactor%20and%20Initial%20Model%20Idea.ipynb) | Reshapes winner/loser box scores into a **team-perspective dataset** (team A vs opponent B, with A's and the opponent's stats and a `win` target), saved as [`transformedTeamsDF.csv`](transformedTeamsDF.csv). Trains Random Forest classifiers on it. |

## Approach and results so far

- **Data refactor:** every game appears twice, once from each team's perspective. Shooting stats (FGM etc.) are excluded from the model
  features because they leak the result. Defensive and rebounding features (`dreb`, `blocks`, `steal`, `fouls`, opponent equivalents, overtime, location) are used instead.
- **Per-team models:** a Random Forest per team (371 teams × 100 random splits) reaches a **median holdout accuracy of 74%** (median weighted F1 0.73).
- **Known limitation:** training accuracy is 1.0, so the per-team forests overfit. Next steps are a single pooled model with
  team-strength features (Elo, Massey ordinals, efficiency margins), probability calibration and Brier-score evaluation (the competition metric).

## Tech

Python · pandas · scikit-learn · seaborn / matplotlib

## Data

Kaggle *March Machine Learning Mania 2025* (`MRegularSeasonDetailedResults.csv`, `MTeams.csv`, `MMasseyOrdinals.csv`, …).
Download it from the competition page into the repo folder before running the notebooks; the raw files are not redistributed here.
