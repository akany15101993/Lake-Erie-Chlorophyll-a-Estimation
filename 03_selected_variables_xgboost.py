"""
Script: 03_selected_variables_xgboost.py
Purpose: XGBoost regression using the fixed predictor set retained after
         the feature-selection stage.

Repository keywords/tags:
# Lake-Erie #chlorophyll-a #Sentinel-3 #OLCI #remote-sensing
# water-quality #XGBoost #nested-cross-validation #fixed-features
# model-validation #reproducible-research

Input:
    Excel workbooks in INPUT_DIR. Each workbook must contain "train" and
    "test" worksheets created by 01_prepare_train_test_data.py.

Target:
    CHA

Fixed predictors supplied in the analysis code (9 predictors):
    Band_12, SST, NDCI, FLH, Band_02, Band_04, Solar_azimuth,
    Solar_zenith, PAR

The independent test set is reserved for final evaluation. Hyperparameter
selection is performed using cross-validation on the training partition.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from scipy.stats import randint, loguniform, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
INPUT_DIR = Path(r"E:\WQ_Erie_lake\Temporal\Data_filling\Model data")
OUTPUT_DIR = Path("XGB_Fixed_Features")
TARGET = "CHA"
RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4
N_ITER = 2000

# IMPORTANT: This list contains nine predictors as supplied.
SELECTED_FEATURES = [
    "Band_12",
    "SST",
    "NDCI",
    "FLH",
    "Band_02",
    "Band_04",
    "Solar_azimuth",
    "Solar_zenith",
    "PAR"
]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(300, 400),
    "max_depth": randint(5, 6),
    "learning_rate": loguniform(0.03, 0.05),
    "subsample": uniform(0.7, 0.2),
    "colsample_bytree": uniform(0.6, 0.25),
    "gamma": uniform(3, 12),
    "min_child_weight": randint(5, 12),
    "reg_alpha": loguniform(0.1, 3),
    "reg_lambda": loguniform(5, 15)
}


def make_model():
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )


def make_search(cv, seed):
    return RandomizedSearchCV(
        estimator=make_model(),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        refit=True,
        verbose=0
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workbooks = sorted(
        p for p in INPUT_DIR.glob("*.xlsx")
        if not p.name.startswith("~$")
    )
    if not workbooks:
        raise FileNotFoundError(f"No .xlsx files found in: {INPUT_DIR}")

    for workbook in workbooks:
        start_time = time.time()
        print(f"\nProcessing: {workbook.name}")

        train_df = pd.read_excel(workbook, sheet_name="train")
        test_df = pd.read_excel(workbook, sheet_name="test")

        required = SELECTED_FEATURES + [TARGET]
        missing_train = [c for c in required if c not in train_df.columns]
        missing_test = [c for c in required if c not in test_df.columns]
        if missing_train or missing_test:
            raise ValueError(
                f"{workbook.name}: missing training columns {missing_train}; "
                f"missing testing columns {missing_test}"
            )

        train_df = train_df.dropna(subset=required).copy()
        test_df = test_df.dropna(subset=required).copy()
        if train_df.empty or test_df.empty:
            raise ValueError(f"{workbook.name}: no complete train or test rows.")

        X_train = train_df[SELECTED_FEATURES].to_numpy()
        y_train = train_df[TARGET].to_numpy()
        X_test = test_df[SELECTED_FEATURES].to_numpy()
        y_test = test_df[TARGET].to_numpy()

        # Nested cross-validation on the training partition.
        outer_cv = KFold(
            n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        outer_rmse, outer_r2 = [], []

        for fold, (tr_idx, val_idx) in enumerate(outer_cv.split(X_train), 1):
            print(f"  Outer fold {fold}/{OUTER_SPLITS}")
            inner_cv = KFold(
                n_splits=INNER_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE + fold
            )
            search = make_search(inner_cv, RANDOM_STATE + fold)
            search.fit(X_train[tr_idx], y_train[tr_idx])

            validation_prediction = search.best_estimator_.predict(X_train[val_idx])
            outer_rmse.append(rmse(y_train[val_idx], validation_prediction))
            outer_r2.append(r2_score(y_train[val_idx], validation_prediction))

        # Tune on all training observations and refit the selected model.
        final_inner_cv = KFold(
            n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        final_search = make_search(final_inner_cv, RANDOM_STATE)
        final_search.fit(X_train, y_train)
        final_model = final_search.best_estimator_

        train_prediction = final_model.predict(X_train)
        test_prediction = final_model.predict(X_test)

        performance = pd.DataFrame([
            {
                "Dataset": "Training",
                "RMSE": rmse(y_train, train_prediction),
                "R2": r2_score(y_train, train_prediction)
            },
            {
                "Dataset": "Independent testing",
                "RMSE": rmse(y_test, test_prediction),
                "R2": r2_score(y_test, test_prediction)
            },
            {
                "Dataset": "Nested CV (mean of outer folds)",
                "RMSE": float(np.mean(outer_rmse)),
                "R2": float(np.mean(outer_r2))
            }
        ])

        cv_summary = pd.DataFrame({
            "Metric": ["RMSE", "R2"],
            "Mean": [np.mean(outer_rmse), np.mean(outer_r2)],
            "Standard deviation": [np.std(outer_rmse), np.std(outer_r2)]
        })

        train_results = pd.DataFrame({
            "Observed": y_train, "Predicted": train_prediction
        })
        test_results = pd.DataFrame({
            "Observed": y_test, "Predicted": test_prediction
        })
        hyperparameters = pd.DataFrame([final_search.best_params_])
        run_info = pd.DataFrame([{
            "Input workbook": workbook.name,
            "Target": TARGET,
            "Number of fixed predictors": len(SELECTED_FEATURES),
            "Predictors": ", ".join(SELECTED_FEATURES),
            "Outer folds": OUTER_SPLITS,
            "Inner folds": INNER_SPLITS,
            "RandomizedSearchCV iterations": N_ITER,
            "Random state": RANDOM_STATE,
            "Processing time (seconds)": time.time() - start_time
        }])

        output_subdir = OUTPUT_DIR / workbook.stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"{workbook.stem}_fixed_features_XGB.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            performance.to_excel(writer, sheet_name="Performance", index=False)
            cv_summary.to_excel(writer, sheet_name="CV Summary", index=False)
            hyperparameters.to_excel(writer, sheet_name="Hyperparameters", index=False)
            train_results.to_excel(writer, sheet_name="Train Predictions", index=False)
            test_results.to_excel(writer, sheet_name="Test Predictions", index=False)
            run_info.to_excel(writer, sheet_name="Run Information", index=False)

        print(performance)
        print(f"Saved: {output_file}")

    print("\nFixed-feature XGBoost workflow completed.")


if __name__ == "__main__":
    main()
