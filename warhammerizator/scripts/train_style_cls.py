import argparse
from pathlib import Path


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path, help="path to training settings")

    return parser.parse_args()


def main():
    args = parse_command_prompt()
    print("Hello world!")


if __name__ == "__main__":
    main()
