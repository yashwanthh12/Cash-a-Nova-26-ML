# Shipping Delay Prediction using Machine Learning

A Random Forest regression project that predicts order shipping delay in days.

## Objective

Predict shipping delay using order and shipment information, assuming the
shipment mode has already been assigned. Because `Shipment Mode` is an input,
this is **not yet a model that chooses the best shipping mode**.

## Dataset

Put your file in this folder and name it `Dataset.csv`. The dataset needs
`Days for shipment (scheduled)` and either:
- an existing `Shipping Delay` target, or
- `Days for shipping (real)` so the target can be calculated.

Target definition:

`Shipping Delay = Days for shipping (real) - Days for shipment (scheduled)`

Negative = early, zero = on time, positive = late.

The dataset is not included in this repository. Do not publish customer or
otherwise sensitive data to a public GitHub repository.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run

```bash
python train_model.py
```

The script cleans text categories, creates `scheduling_pressure`, excludes
post-delivery/leakage columns, imputes missing values, one-hot encodes
categorical columns, trains a Random Forest, and reports test-set MAE and R².

## Outputs

- `shipping_delay_model.joblib`: fitted preprocessing + model pipeline
- `feature_importance.csv`: permutation importance on the held-out test set

Generated outputs and the dataset are excluded from Git by `.gitignore`.

## Interpreting the metrics

- **MAE (days):** average absolute prediction error in days.
- **R²:** compares predictions against a mean-prediction baseline; higher is
  better, but interpret it alongside MAE and a baseline.

Your results may differ from earlier experiments because preprocessing and
feature encoding differ. Use the metrics printed by your final run in your CV
or presentation.

## Important modelling note

`Days for shipping (real)` and `Late_delivery_risk` are excluded because they
reveal or are derived from the delivery outcome. `Shipment Mode` remains an
input, so predictions are conditional on that mode being assigned. A true
mode-selection engine needs a separate design that estimates outcomes for
candidate modes without using the already-selected mode as a predictor.

## Possible next steps

- Compare with a `DummyRegressor` baseline.
- Evaluate with cross-validation.
- Inspect errors by shipment mode and region.
- Build a separate candidate-mode comparison/routing layer.
