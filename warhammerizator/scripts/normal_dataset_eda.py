import argparse
import json
from pathlib import Path
from pprint import pprint
from typing import Dict

import numpy as np
from razdel import sentenize

from warhammerizator.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to normal dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    train_stats = processing_dataset_part(args.input / "train.txt")
    print("Statistics calculation for train part of dataset is finished.")
    val_stats = processing_dataset_part(args.input / "val.txt")
    print("Statistics calculation for val part of dataset is finished.")
    test_stats = processing_dataset_part(args.input / "test.txt")
    print("Statistics calculation for test part of dataset is finished.")
    print()

    save_stats(train_stats, args.output / "train_stats.json")
    save_stats(val_stats, args.output / "val_stats.json")
    save_stats(test_stats, args.output / "test_stats.json")

    print("Train stats:")
    pprint(train_stats, sort_dicts=False)
    print()

    print("Val stats:")
    pprint(val_stats, sort_dicts=False)
    print()

    print("Test stats:")
    pprint(test_stats, sort_dicts=False)
    print()

    print("Done.")


def processing_dataset_part(filename: Path) -> Dict[str, Dict[str, float | int] | int]:
    num_sentences = list()
    num_words = list()
    num_characters = list()
    samples_counter = 0

    with open(filename, "r") as fp:
        for line in fp:
            clean_line = line.strip()

            num_sentences.append(len(list(sentenize(clean_line))))
            num_words.append(len(clean_line.split(" ")))
            num_characters.append(len(clean_line))

            samples_counter += 1

    return {
        "samples": samples_counter,
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


def save_stats(stats: Dict[str, Dict[str, float | int] | int], filename: Path):
    with open(filename, "w") as fp:
        json.dump(obj=stats, fp=fp, indent=4)


if __name__ == "__main__":
    main()
