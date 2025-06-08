import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from razdel import sentenize

from warhammerizator.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    train_stats = processing_dataset_part(args.input / "train.txt")
    val_stats = processing_dataset_part(args.input / "val.txt")
    test_stats = processing_dataset_part(args.input / "test.txt")

    save_stats(train_stats, args.output / "train_stats.json")
    save_stats(val_stats, args.output / "val_stats.json")
    save_stats(test_stats, args.output / "test_stats.json")

    print("Done.")


def processing_dataset_part(filename: Path) -> Dict[str, Dict[str, float | int]]:
    num_sentences = list()
    num_words = list()
    num_characters = list()

    with open(filename, "r") as fp:
        for line in fp:
            clean_line = line.strip()
            if len(clean_line) == 0:
                continue

            num_sentences.append(len(list(sentenize(clean_line))))
            num_words.append(len(clean_line.split(" ")))
            num_characters.append(len(clean_line))

    return {
        "sentence": {
            "mean": float(np.mean(num_sentences)),
            "std": float(np.std(num_sentences)),
            "min": int(np.min(num_sentences)),
            "max": int(np.max(num_sentences))
        },
        "word": {
            "mean": float(np.mean(num_words)),
            "std": float(np.std(num_words)),
            "min": int(np.min(num_words)),
            "max": int(np.max(num_words))
        },
        "character": {
            "mean": float(np.mean(num_characters)),
            "std": float(np.std(num_characters)),
            "min": int(np.min(num_characters)),
            "max": int(np.max(num_characters))
        }
    }


def save_stats(stats: Dict[str, Dict[str, float | int]], filename: Path):
    with open(filename, "w") as fp:
        json.dump(obj=stats, fp=fp, indent=4)


if __name__ == "__main__":
    main()
