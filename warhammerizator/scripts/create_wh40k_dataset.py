import argparse
import random
from pathlib import Path
from typing import List, Tuple

from razdel import sentenize
from sklearn.model_selection import train_test_split

from warhammerizator import conf
from warhammerizator.libs import helpers


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_books", required=True, type=Path, help="path to parsed books")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    parser.add_argument("--num_samples", type=int, default=1000, help="how many samples should generate")
    parser.add_argument("--min_sentences_in_sample", type=int, default=1, help="minimal sentences in sample")
    parser.add_argument("--max_sentences_in_sample", type=int, default=8, help="max sentences in sample")
    parser.add_argument("--min_sequence_len", type=int, default=64, help="min sequence length")
    parser.add_argument("--max_sequence_len", type=int, default=255, help="max sequence length")

    parser.add_argument("--val_size", type=float, default=0.2, help="size of validation part of dataset")
    parser.add_argument("--test_size", type=float, default=0.2, help="size of test part of dataset")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    helpers.create_folder_with_dialog(args.output)

    random.seed(conf.RANDOM_STATE)

    content = read_parsed_books(args.parsed_books)
    samples = generate_samples(
        content,
        args.num_samples,
        args.min_sentences_in_sample,
        args.max_sentences_in_sample,
        args.min_sequence_len,
        args.max_sequence_len
    )
    train, val, test = split_dataset(samples, args.val_size, args.test_size)

    save_dataset_part("train", train, args.output)
    save_dataset_part("val", val, args.output)
    save_dataset_part("test", test, args.output)

    print("Done.")


def read_parsed_books(path: Path) -> List[str]:
    books = list(path.glob("*.txt"))
    books.sort()
    random.shuffle(books)

    content = list()
    for book_filename in books:
        with open(book_filename, "r") as fp:
            text = fp.read()
            content.extend([sentence.text for sentence in sentenize(text)])

    return content


def generate_samples(
        content: List[str],
        num_samples: int,
        min_sentences_in_sample: int,
        max_sentences_in_sample: int,
        min_sequence_len: int,
        max_sequence_len: int
) -> List[str]:
    used_start_pos = set()
    result = list()

    while len(result) < num_samples:
        start_pos = random.randrange(0, len(content))

        if start_pos in used_start_pos:
            continue

        num_sentences = random.randint(min_sentences_in_sample, max_sentences_in_sample)

        sentences = [content[start_pos + i] for i in range(num_sentences)]
        sample = " ".join(sentences)
        if min_sequence_len <= len(sample) <= max_sequence_len:
            result.append(sample)
            used_start_pos.add(start_pos)

    return result


def split_dataset(samples: List[str], val_size: float, test_size: float) -> Tuple[List[str], List[str], List[str]]:
    tmp, test = train_test_split(samples, test_size=test_size, random_state=conf.RANDOM_STATE, shuffle=True)

    new_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(tmp, test_size=new_val_size, random_state=conf.RANDOM_STATE, shuffle=True)

    return train, val, test


def save_dataset_part(part: str, samples: List[str], output_path: Path) -> None:
    with open(output_path / f"{part}.txt", "w") as fp:
        for sample in samples:
            fp.write(f"{sample}\n")


if __name__ == "__main__":
    main()
