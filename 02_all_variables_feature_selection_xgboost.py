"""
Script: 02_all_variables_feature_selection_xgboost.py
Purpose: Feature selection from the full candidate predictor set using
         ElasticNet-based SHAP importance and correlation pruning, followed
         by XGBoost regression with nested cross-validation.

Repository keywords/tags:
# Lake-Erie #chlorophyll-a #Sentinel-3 #OLCI #remote-sensing
# explainable-AI #SHAP #ElasticNet #feature-selection #XGBoost
# nested-cross-validation #water-quality

Input:
    Excel workbooks in INPUT_DIR. Each workbook must contain "train" and
    "test" worksheets created by 01_prepare_train_test_data.py.

Target:
    CHA

Feature-selection procedure:
    1. Standardize predictors within each training fold.
    2. Fit ElasticNetCV.
    3. Calculate mean absolute SHAP values for the ElasticNet model.
    4. Iteratively remove the lower-SHAP-importance member of the most
       highly correlated predictor pair until the correlation threshold
       is met or MIN_FEATURES remain.

Important:
    Feature selection is implemented as a scikit-learn transformer inside
    the model pipeline. Consequently, it is refitted within each inner-CV
    training fold rather than being calculated once using the entire
    training dataset. The independent test set is used only for final
    performance evaluation and SHAP reporting is based on the final model.

Note:
    The search is computationally intensive: 2,000 parameter combinations
    are evaluated in each inner search. Adjust N_ITER only if necessary,
    and report the actual value used in the manuscript.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import shap

from scipy.stats import randint, loguniform, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
INPUT_DIR = Path(r"E:\WQ_Erie_lake\Temporal\Data_testing\Model data")
OUTPUT_DIR = Path("XGB_All_Variables")
TARGET = "CHA"
RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4
N_ITER = 2000
CORRELATION_THRESHOLD = 0.90
MIN_FEATURES = 20

# Candidate predictors supplied for the manuscript analysis.
CANDIDATE_FEATURES = [
    "Band_02", "Band_03", "Band_04", "Band_05",
    "Band_06", "Band_07", "Band_08", "Band_09",
    "Band_11", "Band_12", "Band_13", "Band_14", "Band_15",
    "Band_16", "Band_17", "Band_18", "Band_19", "Band_20",
    "Band_21", "Solar_azimuth", "Solar_zenith", "PAR", "SST",
    "NDCI", "FLH", "MCI"
]

# ---------------------------------------------------------------------
# FEATURE SELECTOR
# ---------------------------------------------------------------------
class ElasticNetSHAPCorrelationSelector(BaseEstimator, TransformerMixin):
    """Select predictors using ElasticNet SHAP importance and correlation."""

    def __init__(
        self,
        feature_names,
        correlation_threshold=0.90,
        min_features=20,
        random_state=42
    ):
        self.feature_names = feature_names
        self.correlation_threshold = correlation_threshold
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        X_df = pd.DataFrame(X, columns=self.feature_names)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_df)

        self.elastic_net_ = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5,
            max_iter=10000,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.elastic_net_.fit(X_scaled, np.asarray(y).ravel())

        explainer = shap.Explainer(self.elastic_net_, X_scaled)
        shap_values = explainer(X_scaled)
        importance = pd.Series(
            np.abs(shap_values.values).mean(axis=0),
            index=self.feature_names
        )

        corr = pd.DataFrame(X_scaled, columns=self.feature_names).corr().abs()
        remaining = list(self.feature_names)

        while len(remaining) > self.min_features:
            current_corr = corr.loc[remaining, remaining].to_numpy()
            upper = np.triu(current_corr, k=1)
            if upper.size == 0 or np.nanmax(upper) < self.correlation_threshold:
                break

            i, j = np.unravel_index(np.nanargmax(upper), upper.shape)
            feature_1, feature_2 = remaining[i], remaining[j]
            drop_feature = (
                feature_1 if importance[feature_1] < importance[feature_2]
                else feature_2
            )
            remaining.remove(drop_feature)

        self.selected_features_ = remaining
        self.selected_indices_ = [
            list(self.feature_names).index(name) for name in remaining
        ]
        self.removed_features_ = [
            name for name in self.feature_names if name not in remaining
        ]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


# ---------------------------------------------------------------------
# METRICS AND SEARCH SPACE
# ---------------------------------------------------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


PARAM_DISTRIBUTIONS = {
    "xgb__n_estimators": randint(300, 400),
    "xgb__max_depth": randint(5, 6),
    "xgb__learning_rate": loguniform(0.03, 0.05),
    "xgb__subsample": uniform(0.7, 0.2),
    "xgb__colsample_bytree": uniform(0.6, 0.25),
    "xgb__gamma": uniform(3, 12),
    "xgb__min_child_weight": randint(5, 12),
    "xgb__reg_alpha": loguniform(0.1, 3),
    "xgb__reg_lambda": loguniform(5, 15)
}


def make_pipeline():
    selector = ElasticNetSHAPCorrelationSelector(
        feature_names=CANDIDATE_FEATURES,
        correlation_threshold=CORRELATION_THRESHOLD,
        min_features=MIN_FEATURES,
        random_state=RANDOM_STATE
    )
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )
    return Pipeline([("selector", selector), ("xgb", xgb)])


def make_search(cv, seed):
    return RandomizedSearchCV(
        estimator=make_pipeline(),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=seed,
        refit=True,
        verbose=0
    )


# ---------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------
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

        required = CANDIDATE_FEATURES + [TARGET]
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

        X_train = train_df[CANDIDATE_FEATURES].to_numpy()
        y_train = train_df[TARGET].to_numpy()
        X_test = test_df[CANDIDATE_FEATURES].to_numpy()
        y_test = test_df[TARGET].to_numpy()

        # Outer CV estimates the full feature-selection + tuning workflow.
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

            prediction = search.best_estimator_.predict(X_train[val_idx])
            outer_rmse.append(rmse(y_train[val_idx], prediction))
            outer_r2.append(r2_score(y_train[val_idx], prediction))

        # Tune once on the complete training partition, then refit on it.
        final_inner_cv = KFold(
            n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        final_search = make_search(final_inner_cv, RANDOM_STATE)
        final_search.fit(X_train, y_train)
        final_model = final_search.best_estimator_

        selected_features = final_model.named_steps["selector"].selected_features_
        removed_features = final_model.named_steps["selector"].removed_features_

        train_prediction = final_model.predict(X_train)
        test_prediction = final_model.predict(X_test)

        # SHAP importance for the final XGBoost model, using selected predictors.
        X_train_selected = final_model.named_steps["selector"].transform(X_train)
        xgb_model = final_model.named_steps["xgb"]
        tree_explainer = shap.TreeExplainer(xgb_model)
        shap_values = tree_explainer.shap_values(X_train_selected)
        shap_importance = pd.DataFrame({
            "Feature": selected_features,
            "Mean absolute SHAP value": np.abs(shap_values).mean(axis=0)
        }).sort_values("Mean absolute SHAP value", ascending=False)

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

        predictions_train = pd.DataFrame({
            "Observed": y_train, "Predicted": train_prediction
        })
        predictions_test = pd.DataFrame({
            "Observed": y_test, "Predicted": test_prediction
        })
        features_df = pd.DataFrame({
            "Selected features": pd.Series(selected_features),
            "Removed features": pd.Series(removed_features)
        })
        hyperparameters = pd.DataFrame([final_search.best_params_])
        run_info = pd.DataFrame([{
            "Input workbook": workbook.name,
            "Candidate predictor count": len(CANDIDATE_FEATURES),
            "Selected predictor count": len(selected_features),
            "Outer folds": OUTER_SPLITS,
            "Inner folds": INNER_SPLITS,
            "RandomizedSearchCV iterations": N_ITER,
            "Random state": RANDOM_STATE,
            "Processing time (seconds)": time.time() - start_time
        }])

        output_subdir = OUTPUT_DIR / workbook.stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / f"{workbook.stem}_all_variables_XGB.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            performance.to_excel(writer, sheet_name="Performance", index=False)
            cv_summary.to_excel(writer, sheet_name="CV Summary", index=False)
            features_df.to_excel(writer, sheet_name="Feature Selection", index=False)
            shap_importance.to_excel(writer, sheet_name="SHAP Importance", index=False)
            hyperparameters.to_excel(writer, sheet_name="Hyperparameters", index=False)
            predictions_train.to_excel(writer, sheet_name="Train Predictions", index=False)
            predictions_test.to_excel(writer, sheet_name="Test Predictions", index=False)
            run_info.to_excel(writer, sheet_name="Run Information", index=False)

        print(f"Selected predictors ({len(selected_features)}): {selected_features}")
        print(performance)
        print(f"Saved: {output_file}")

    print("\nAll-variable feature-selection and XGBoost workflow completed.")


if __name__ == "__main__":
    main()
