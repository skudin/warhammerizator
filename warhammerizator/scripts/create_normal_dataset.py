import argparse
import json
from pathlib import Path
from typing import List

from razdel import sentenize

from warhammerizator.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")
    parser.add_argument("--max_sequence_len", type=int, default=127, help="max sequence length")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    train = processing_dataset_part(args.input / "gazeta_train.jsonl", args.max_sequence_len)
    val = processing_dataset_part(args.input / "gazeta_val.jsonl", args.max_sequence_len)
    test = processing_dataset_part(args.input / "gazeta_test.jsonl", args.max_sequence_len)

    save_dataset_part(train, args.output / "train.txt")
    save_dataset_part(val, args.output / "val.txt")
    save_dataset_part(test, args.output / "test.txt")

    print("Done.")


def processing_dataset_part(filename: Path, max_sequence_len: int) -> List[str]:
    result = list()

    with open(filename, "r") as fp:
        for line in fp:
            data = json.loads(line)
            if len(data["summary"]) <= max_sequence_len:
                result.append(data["summary"])
            else:
                summary_sentences = list(sentenize(data["summary"]))

                sample = ""
                for sentence in summary_sentences:
                    if len(sample) + len(sentence.text) < max_sequence_len:
                        sample += sentence.text
                    else:
                        break

                if sample != "":
                    result.append(sample)

    return result


def save_dataset_part(dataset_part: List[str], filename: Path) -> None:
    with open(filename, "w") as fp:
        for item in dataset_part:
            fp.write(f"{item}\n")


if __name__ == "__main__":
    main()
