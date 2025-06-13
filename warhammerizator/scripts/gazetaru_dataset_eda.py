import argparse
import json
from pathlib import Path
from typing import Dict

from warhammerizator.libs.eda_base import EDABase


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="path to gazeta ru dataset")
    parser.add_argument("--output", required=True, type=Path, help="output path")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    dataset_parts = {
        "train": "gazeta_train.jsonl",
        "val": "gazeta_val.jsonl",
        "test": "gazeta_test.jsonl"
    }

    eda = GazetaRuDatasetEDA(args.input, ("summary", "text"), dataset_parts)
    stats = eda.processing()
    eda.save_stats(stats, args.output)
    eda.print_stats(stats)

    print("Done.")


class GazetaRuDatasetEDA(EDABase):
    def _processing_line(self, line: str) -> Dict[str, str]:
        data = json.loads(line)
        return {
            "summary": data["summary"],
            "text": data["text"]
        }


if __name__ == "__main__":
    main()
