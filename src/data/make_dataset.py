import logging
from pathlib import Path

import click
import pandas as pd
import yaml

COLUMN_NAMES = (
    ["unit_id", "cycle", "setting1", "setting2", "setting3"]
    + [f"sensor{i}" for i in range(1, 22)]
)


def load_config(config_path="configs/base.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_txt(path: str) -> pd.DataFrame:
    """Load a space-delimited C-MAPSS file (train or test), no header."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]  # drop any trailing empty columns from stray spaces
    df.columns = COLUMN_NAMES
    return df


def add_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    """For training data: RUL = max cycle for that engine - current cycle.
    The engine runs to failure, so max cycle IS the failure point."""
    df = df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def add_rul_test(df: pd.DataFrame, rul_path: str) -> pd.DataFrame:
    """For test data: RUL at truncation = true_RUL (given) + engine's max
    cycle in this file - current cycle. This accounts for the gap between
    where the test trajectory was cut off and the true failure point."""
    df = df.copy()
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_RUL"])
    true_rul["unit_id"] = true_rul.index + 1  # RUL file is ordered by unit_id, 1-indexed

    max_cycle = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycle.columns = ["unit_id", "max_cycle"]

    df = df.merge(max_cycle, on="unit_id").merge(true_rul, on="unit_id")
    df["RUL"] = df["true_RUL"] + df["max_cycle"] - df["cycle"]
    df = df.drop(columns=["max_cycle", "true_RUL"])
    return df


@click.command()
@click.option('--config-path', default='configs/base.yaml', help='Path to config YAML')
@click.option('--dataset', default='FD001', help='Which C-MAPSS sub-dataset to use (FD001-FD004)')
def main(config_path, dataset):
    """Load raw C-MAPSS train/test files, assign column names, compute RUL labels,
    and save cleaned versions to data/processed/."""
    logger = logging.getLogger("make_dataset")

    raw_dir = Path("data/raw") / dataset
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {dataset} train data...")
    train_df = load_raw_txt(raw_dir / f"train_{dataset}.txt")
    train_df = add_rul_train(train_df)
    logger.info(f"Train shape: {train_df.shape}, engines: {train_df['unit_id'].nunique()}")

    logger.info(f"Loading {dataset} test data...")
    test_df = load_raw_txt(raw_dir / f"test_{dataset}.txt")
    test_df = add_rul_test(test_df, raw_dir / f"RUL_{dataset}.txt")
    logger.info(f"Test shape: {test_df.shape}, engines: {test_df['unit_id'].nunique()}")

    train_out = processed_dir / f"{dataset}_train_cleaned.csv"
    test_out = processed_dir / f"{dataset}_test_cleaned.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.info(f"Saved cleaned train data to {train_out}")
    logger.info(f"Saved cleaned test data to {test_out}")
    logger.info(f"Train RUL range: {train_df['RUL'].min()} to {train_df['RUL'].max()}")
    logger.info(f"Test RUL range: {test_df['RUL'].min()} to {test_df['RUL'].max()}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format=log_fmt)
    main()