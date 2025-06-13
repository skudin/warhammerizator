import argparse
from pathlib import Path
from typing import Dict

from warhammerizator.libs.eda_base import EDABase


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    dataset_parts = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt"
    }
    eda = WH40kDatasetEDA(args.input, ("text", ), dataset_parts)
    stats = eda.processing()
    eda.save_stats(stats, args.output)
    eda.print_stats(stats)

    print("Done.")


class WH40kDatasetEDA(EDABase):
    def _processing_line(self, line: str) -> Dict[str, str]:
        return {
            "text": line.strip()
        }


if __name__ == "__main__":
    main()
