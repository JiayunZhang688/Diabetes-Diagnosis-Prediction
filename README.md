# Lab 23 – Group 10: Diabetes Diagnosis Prediction

Predicting whether an individual is diabetic from demographic and lifestyle
factors (BMI, age, income, general health, smoking status, physical
activity, etc.). The project is split into two notebooks that run in
sequence.

## Files

| File | Purpose |
|---|---|
| `diabetes_diagnosis_stage1.ipynb` | EDA: problem definition, data description, cleaning, missing-value handling, exploratory plots. Outputs `diabetes_diagnosis_cleaned.csv`. |
| `diabetes_diagnosis_stage2.ipynb` | Modelling: feature screening (Income), preprocessing, and three model families — logistic regression, random forest, decision tree — with class-imbalance handling (class weights, SMOTE, threshold tuning) and hyperparameter search. |

## How to run

1. Place the raw dataset as `diabetes_diagnosis.csv` in this folder (not
   included in this handoff — pull it from wherever your group originally
   sourced it, e.g. the course Ed/Canvas data folder or your shared drive).
2. Run `diabetes_diagnosis_stage1.ipynb` top to bottom. This produces
   `diabetes_diagnosis_cleaned.csv` in the same folder.
3. Run `diabetes_diagnosis_stage2.ipynb` top to bottom (it reads the cleaned
   CSV from step 2).

Both notebooks assume `scikit-learn`, `imbalanced-learn`, `pandas`,
`numpy`, `seaborn`, `matplotlib`, and `scipy` are installed.

## What was fixed

`diabetes_diagnosis_stage2.ipynb` previously stopped at the very first
model (baseline logistic regression) with:

```
InvalidParameterError: The 'penalty' parameter of LogisticRegression must
be a str among {'l2', 'elasticnet', 'l1'} or None. Got 'none' instead.
```

Cause: `penalty='none'` was passed as the *string* `'none'` in two cells
(baseline logistic regression, and the class-weight-balanced logistic
regression). Recent scikit-learn versions require the literal Python value
`None` (no quotes) to mean "no penalty." Both occurrences were changed from
`penalty='none'` to `penalty=None`.

Everything downstream of that point — the rest of logistic regression
(L1/L2, GridSearchCV), random forest (baseline, more complex, SMOTE,
threshold tuning), and decision tree (baseline, balanced, GridSearchCV,
best-model evaluation, feature importance) — had never actually been
executed in the submitted file (all cells showed no output / no execution
count). I ran the corrected code end-to-end against a synthetic dataset
built with the same column names and types the notebook expects, and it
completes without further errors — so as far as code correctness goes, the
rest of the pipeline runs fine. I could not verify it against your real
data or sanity-check whether the resulting metrics/plots look reasonable,
since the raw CSV wasn't in the handoff — worth doing that pass yourself
once you re-run it locally.

One thing to keep an eye on when you re-run: `RocCurveDisplay` raises a
`FutureWarning` on newer scikit-learn about passing plot styling as
`**kwargs` instead of a `curve_kwargs` dict — harmless for now, but may
break in a future scikit-learn release if you don't update it.

## Notebook structure at a glance

**Stage 1 (EDA)**
- Step 1: Problem definition
- Step 2: Data description (shape, dtypes, missingness, categorical levels)
- Step 3: Cleaning (issue identification, cleaning strategy, missing-value
  handling)
- Step 4: Exploratory analysis (BMI/income vs. diabetes correlation,
  distribution plots, diabetes rate by general health/education)

**Stage 2 (Modelling)**
- Feature screening: Income dropped after point-biserial / Spearman
  correlation and Mann–Whitney U test showed a weak relationship with the
  target
- Preprocessing: `StandardScaler` for numeric features, `OneHotEncoder`
  for categorical features, 2/3–1/3 stratified train/test split
- Logistic regression: baseline, class-weight balanced, L2/L1 regularised,
  GridSearchCV over `C`, custom label-weight tuning
- Random forest: baseline, increased complexity, SMOTE, classification
  threshold tuning
- Decision tree: baseline, class-weight balanced, GridSearchCV
  hyperparameter tuning, best-model evaluation, feature importance
