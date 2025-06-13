import argparse
import json
import random
from pathlib import Path
from typing import List

from razdel import sentenize

from warhammerizator import conf
from warhammerizator.libs import helpers
from warhammerizator.libs import data_tools


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    parser.add_argument("--min_sentences_in_sample", type=int, default=1, help="minimal sentences in sample")
    parser.add_argument("--max_sentences_in_sample", type=int, default=8, help="max sentences in sample")
    parser.add_argument("--min_sequence_len", type=int, default=64, help="min sequence length")
    parser.add_argument("--max_sequence_len", type=int, default=127, help="max sequence length")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    random.seed(conf.RANDOM_STATE)

    print("Processing train part.")
    train = processing_dataset_part(
        args.input / "gazeta_train.jsonl",
        args.min_sentences_in_sample,
        args.max_sentences_in_sample,
        args.min_sequence_len,
        args.max_sequence_len
    )

    print("Processing val part.")
    val = processing_dataset_part(
        args.input / "gazeta_val.jsonl",
        args.min_sentences_in_sample,
        args.max_sentences_in_sample,
        args.min_sequence_len,
        args.max_sequence_len
    )

    print("Processing test part.")
    test = processing_dataset_part(
        args.input / "gazeta_test.jsonl",
        args.min_sentences_in_sample,
        args.max_sentences_in_sample,
        args.min_sequence_len,
        args.max_sequence_len
    )

    save_dataset_part(train, args.output / "train.txt")
    save_dataset_part(val, args.output / "val.txt")
    save_dataset_part(test, args.output / "test.txt")

    print("Done.")


def processing_dataset_part(
        filename: Path,
        min_sentences_in_sample: int,
        max_sentences_in_sample: int,
        min_sequence_len: int,
        max_sequence_len: int
) -> List[str]:
    result = list()

    with open(filename, "r") as fp:
        for num, line in enumerate(fp):
            item = json.loads(line)
            content = [sentence.text.strip() for sentence in sentenize(item["text"])]

            sample = data_tools.generate_samples(
                content=content,
                num_samples=1,
                min_sentences_in_sample=min_sentences_in_sample,
                max_sentences_in_sample=max_sentences_in_sample,
                min_sequence_len=min_sequence_len,
                max_sequence_len=max_sequence_len
            )

            if len(sample) > 0:
                result.append(sample[0])
                print(f"Added item number {num + 1}.")

    return result


def save_dataset_part(dataset_part: List[str], filename: Path) -> None:
    with open(filename, "w") as fp:
        for item in dataset_part:
            fp.write(f"{item}\n")


if __name__ == "__main__":
    main()
