import logging
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def load_config(config_path="configs/base.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def phm08_score(y_true, y_pred):
    """PHM08 competition asymmetric scoring function.

    Penalizes LATE predictions (predicting more remaining life than actually
    exists) far more heavily than EARLY predictions, since underestimating
    remaining life just means slightly early maintenance (safe), while
    overestimating it risks an in-flight failure (catastrophic).

    d = predicted - actual
    d < 0 (early/safe):  score = exp(-d/13)  - 1   (gentle penalty)
    d >= 0 (late/risky): score = exp( d/10)  - 1   (steep penalty)

    Lower total score is better. This is the exact formula used in the
    original PHM08 data challenge for this dataset.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def evaluate_model(name, y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    phm08 = phm08_score(y_true, y_pred)
    return {"model": name, "rmse": rmse, "mae": mae, "phm08_score": phm08}


@click.command()
@click.option('--config-path', default='configs/base.yaml', help='Path to config YAML')
@click.option('--dataset', default='FD001', help='Which C-MAPSS sub-dataset to use')
def main(config_path, dataset):
    """Train Random Forest and XGBoost regressors to predict RUL, compare, save the best."""
    logger = logging.getLogger("train_model")
    config = load_config(config_path)
    random_state = config.get("random_state", 42)

    processed_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {dataset} feature data...")
    train_df = pd.read_csv(processed_dir / f"{dataset}_train_features.csv")
    test_df = pd.read_csv(processed_dir / f"{dataset}_test_features.csv")
    feature_cols = joblib.load(processed_dir / f"{dataset}_feature_cols.joblib")

    # For test evaluation: use only the LAST cycle per engine, since that's
    # the point at which we'd actually be making a real prediction (the
    # "current" state of each engine in the test set).
    test_last = test_df.sort_values("cycle").groupby("unit_id").tail(1)

    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]
    X_test = test_last[feature_cols]
    y_test = test_last["RUL"]

    logger.info(f"Train: {X_train.shape}, Test (last cycle per engine): {X_test.shape}")

    results = []
    models = {}

    logger.info("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results.append(evaluate_model("RandomForest", y_test, rf_preds))
    models["RandomForest"] = rf

    logger.info("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=random_state)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    results.append(evaluate_model("XGBoost", y_test, xgb_preds))
    models["XGBoost"] = xgb

    results_df = pd.DataFrame(results).set_index("model")
    logger.info("\n" + results_df.to_string())
    results_df.to_csv(models_dir / f"{dataset}_model_comparison.csv")

    best_name = results_df["rmse"].idxmin()
    best_model = models[best_name]

    joblib.dump(best_model, models_dir / f"{dataset}_best_model.joblib")
    logger.info(f"Best model (lowest RMSE): {best_name}")
    logger.info(f"  RMSE={results_df.loc[best_name, 'rmse']:.2f} cycles, "
                f"MAE={results_df.loc[best_name, 'mae']:.2f} cycles, "
                f"PHM08 score={results_df.loc[best_name, 'phm08_score']:.1f}")
    logger.info(f"Saved best model to {models_dir / f'{dataset}_best_model.joblib'}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format=log_fmt)
    main()