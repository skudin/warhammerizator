import argparse
from pathlib import Path
from typing import Dict, Tuple

import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path, help="path to training settings")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    settings = read_settings(args.settings)

    train, val, test, le = prepare_dataset(settings["dataset"])

    print("Hello world!")


def read_settings(filename: Path) -> Dict:
    with open(filename, "r") as fp:
        return yaml.load(stream=fp, Loader=yaml.FullLoader)


def prepare_dataset(dataset_paths: Dict[str, str]) -> Tuple[Dataset, Dataset, Dataset, LabelEncoder]:
    train_df = pd.read_csv(dataset_paths["train"])
    val_df = pd.read_csv(dataset_paths["val"])
    test_df = pd.read_csv(dataset_paths["test"])

    le = LabelEncoder()
    train_df.rate = le.fit_transform(train_df.label)

    train = Dataset.from_pandas(train_df)
    val = Dataset.from_pandas(val_df)
    test = Dataset.from_pandas(test_df)

    return train, val, test, le


if __name__ == "__main__":
    main()
