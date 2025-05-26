import argparse
from pathlib import Path
from typing import Dict

import yaml


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path, help="path to training settings")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    settings = read_settings(args.settings)

    print("Hello world!")


def read_settings(filename: Path) -> Dict:
    with open(filename, "r") as fp:
        return yaml.load(stream=fp, Loader=yaml.FullLoader)



if __name__ == "__main__":
    main()
