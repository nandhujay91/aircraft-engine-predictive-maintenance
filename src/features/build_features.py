import logging
from pathlib import Path

import click
import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

CONSTANT_SENSORS = ["sensor1", "sensor5", "sensor10", "sensor16", "sensor18", "sensor19"]
ROLLING_WINDOW = 5
RUL_CLIP = 125


def load_config(config_path="configs/base.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sensors that never change -- they add no predictive signal."""
    cols_to_drop = [c for c in CONSTANT_SENSORS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add rolling mean and std for each sensor, computed per engine (unit_id),
    since a single noisy reading is far less predictive than the recent trend."""
    df = df.sort_values(["unit_id", "cycle"]).copy()
    sensor_cols = [c for c in df.columns if c.startswith("sensor")]

    grouped = df.groupby("unit_id")[sensor_cols]
    rolling_mean = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
    rolling_std = grouped.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))

    rolling_mean.columns = [f"{c}_roll_mean" for c in sensor_cols]
    rolling_std.columns = [f"{c}_roll_std" for c in sensor_cols]

    df = pd.concat([df, rolling_mean, rolling_std], axis=1)
    return df


def clip_rul(df: pd.DataFrame, clip_value: int = RUL_CLIP) -> pd.DataFrame:
    """Cap RUL at a maximum value. Standard practice in C-MAPSS literature --
    the difference between 300 and 350 cycles remaining isn't meaningfully
    actionable, and unclipped RUL destabilizes training."""
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=clip_value)
    return df


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Full feature pipeline: drop constants, add rolling stats, scale, clip RUL.
    Scaler is fit on train only, applied to both -- standard practice to avoid leakage."""
    train_df = drop_constant_sensors(train_df)
    test_df = drop_constant_sensors(test_df)

    train_df = add_rolling_features(train_df)
    test_df = add_rolling_features(test_df)

    train_df = clip_rul(train_df)
    test_df = clip_rul(test_df)

    feature_cols = [c for c in train_df.columns if c not in ("unit_id", "cycle", "RUL")]

    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler, feature_cols


@click.command()
@click.option('--config-path', default='configs/base.yaml', help='Path to config YAML')
@click.option('--dataset', default='FD001', help='Which C-MAPSS sub-dataset to use')
def main(config_path, dataset):
    """Build model-ready features from cleaned C-MAPSS data."""
    logger = logging.getLogger("build_features")

    processed_dir = Path("data/processed")

    logger.info(f"Loading cleaned {dataset} data...")
    train_df = pd.read_csv(processed_dir / f"{dataset}_train_cleaned.csv")
    test_df = pd.read_csv(processed_dir / f"{dataset}_test_cleaned.csv")

    logger.info("Building features (dropping constants, rolling stats, scaling, RUL clipping)...")
    train_df, test_df, scaler, feature_cols = build_features(train_df, test_df)

    logger.info(f"Final feature count: {len(feature_cols)}")
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    train_out = processed_dir / f"{dataset}_train_features.csv"
    test_out = processed_dir / f"{dataset}_test_features.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    joblib.dump(scaler, processed_dir / f"{dataset}_scaler.joblib")
    joblib.dump(feature_cols, processed_dir / f"{dataset}_feature_cols.joblib")

    logger.info(f"Saved train features to {train_out}")
    logger.info(f"Saved test features to {test_out}")
    logger.info(f"Saved scaler and feature column list to {processed_dir}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format=log_fmt)
    main()