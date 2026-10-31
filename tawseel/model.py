
"""Failure-risk baseline for Tawseel COD parcels."""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    FunctionTransformer,
    StandardScaler,
)
TARGET = "failed"

FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer history using strictly earlier parcels only."""
    required = {"customer_id", "created_at", "failed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    history = (
        out.groupby(
            ["customer_id", "created_at"],
            dropna=False,
            sort=True,
        )["failed"]
        .agg(parcel_count="size", failed_count="sum")
        .sort_index()
    )

    grouped = history.groupby(level=0, sort=False)

    history["prior_parcels"] = (
        grouped["parcel_count"].cumsum() - history["parcel_count"]
    )
    history["prior_failures"] = (
        grouped["failed_count"].cumsum() - history["failed_count"]
    )

    row_keys = pd.MultiIndex.from_frame(
        out[["customer_id", "created_at"]]
    )
    aligned = history.reindex(row_keys)

    prior_count = aligned["prior_parcels"].to_numpy(dtype=int)
    prior_failures = aligned["prior_failures"].to_numpy(dtype=float)

    out["customer_prior_parcels"] = prior_count
    out["customer_prior_fail_rate"] = np.divide(
        prior_failures,
        prior_count,
        out=np.full(len(out), np.nan, dtype=float),
        where=prior_count > 0,
    )

    return out


def time_split(
    df: pd.DataFrame,
    time_col: str = "created_at",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically, keeping equal timestamps together."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    if len(df) == 0:
        return df.copy(), df.copy()

    n_test = ceil(len(df) * test_fraction)
    sorted_times = df[time_col].sort_values(kind="stable")
    boundary = sorted_times.iloc[len(df) - n_test]

    train = df.loc[df[time_col] < boundary].copy()
    test = df.loc[df[time_col] >= boundary].copy()

    return train, test


def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Select allowed inputs and extract useful time features."""
    X = X.copy()

    columns = FEATURE_COLUMNS + [
        "customer_prior_parcels",
        "customer_prior_fail_rate",
    ]
    X = X[columns]

    created = pd.to_datetime(X["created_at"], errors="coerce")
    X["created_hour"] = created.dt.hour
    X["created_dayofweek"] = created.dt.dayofweek

    return X.drop(columns=["created_at"])


def build_pipeline(seed: int = 0) -> Pipeline:
    """Build an unfitted classification pipeline."""
    numeric_features = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate",
        "created_hour",
        "created_dayofweek",
    ]

    categorical_features = [
        "customer_id",
        "merchant_id",
        "area",
        "promised_slot",
        "address_text",
    ]

    numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("prepare", FunctionTransformer(_prepare_features, validate=False)),
        ("preprocessing", preprocessing),
        (
            "classifier",
            LogisticRegression(
                max_iter=1500,
                C=1.0,
                random_state=seed,
            ),
        ),
    ])


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Evaluate probability predictions."""
    y_true = test_df[TARGET].to_numpy()
    probabilities = model.predict_proba(test_df)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
        "base_rate": float(np.mean(y_true)),
        "n": len(y_true),
    }


def compare_to_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """Compare the model with a prior-probability dummy baseline."""
    y_train = train_df[TARGET]

    model = build_pipeline(seed)
    model.fit(train_df, y_train)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)

    model_results = evaluate(model, test_df)

    y_test = test_df[TARGET].to_numpy()
    dummy_probabilities = dummy.predict_proba(
        np.zeros((len(test_df), 1))
    )[:, 1]

    dummy_results = {
        "roc_auc": roc_auc_score(y_test, dummy_probabilities),
        "pr_auc": average_precision_score(y_test, dummy_probabilities),
        "brier": brier_score_loss(y_test, dummy_probabilities),
        "base_rate": float(np.mean(y_test)),
        "n": len(y_test),
    }

    return {
        "model": model_results,
        "dummy": dummy_results,
        "lift_pr_auc": model_results["pr_auc"] - dummy_results["pr_auc"],
    }
