import argparse
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from warhammerizator import conf
from warhammerizator.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_dataset", required=True, type=Path, help="path to normal dataset")
    parser.add_argument("--wh40k_dataset", required=True, type=Path, help="path to Warhammer 40k dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    random.seed(conf.RANDOM_STATE)

    normal_train, normal_val, normal_test = read_dataset(args.normal_dataset)
    wh40k_train, wh40k_val, wh40k_test = read_dataset(args.wh40k_dataset)

    train, val, test = create_dataset_for_cls(normal_train, normal_val, normal_test, wh40k_train, wh40k_val, wh40k_test)

    train.to_csv(args.output / "train.csv", index=False)
    val.to_csv(args.output / "val.csv", index=False)
    test.to_csv(args.output / "test.csv", index=False)

    print("Done.")


def read_dataset(path: Path) -> Tuple[List[str], List[str], List[str]]:
    train = read_dataset_part(path / "train.txt")
    val = read_dataset_part(path / "val.txt")
    test = read_dataset_part(path / "test.txt")

    return train, val, test


def read_dataset_part(filename: Path) -> List[str]:
    with open(filename, "r") as fp:
        dataset_part = [line.strip() for line in fp]

    return dataset_part


def create_dataset_for_cls(
        normal_train: List[str],
        normal_val: List[str],
        normal_test: List[str],
        wh40k_train: List[str],
        wh40k_val: List[str],
        wh40k_test: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = create_dataset_part_for_cls(normal_train, wh40k_train)
    val = create_dataset_part_for_cls(normal_val, wh40k_val)
    test = create_dataset_part_for_cls(normal_test, wh40k_test)

    return train, val, test


def create_dataset_part_for_cls(normal_dataset, wh40k_dataset):
    data = [(0, text) for text in normal_dataset]
    data.extend([(1, text) for text in wh40k_dataset])

    random.shuffle(data)

    df = pd.DataFrame(data, columns=("label", "text"))

    return df


if __name__ == "__main__":
    main()
