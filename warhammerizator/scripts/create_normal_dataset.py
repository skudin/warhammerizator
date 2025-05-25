import argparse
import json
from pathlib import Path
from typing import List

from warhammerizator.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    train = processing_dataset_part(args.input / "gazeta_train.jsonl")
    val = processing_dataset_part(args.input / "gazeta_val.jsonl")
    test = processing_dataset_part(args.input / "gazeta_test.jsonl")

    save_dataset_part(train, args.output / "train.txt")
    save_dataset_part(val, args.output / "val.txt")
    save_dataset_part(test, args.output / "test.txt")

    print("Done.")


def processing_dataset_part(filename: Path) -> List[str]:
    result = list()

    with open(filename, "r") as fp:
        for line in fp:
            data = json.loads(line)
            result.append(data["summary"])

    return result


def save_dataset_part(dataset_part: List[str], filename: Path) -> None:
    with open(filename, "w") as fp:
        for item in dataset_part:
            fp.write(f"{item}\n")


if __name__ == "__main__":
    main()
