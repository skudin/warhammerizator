import argparse
from pathlib import Path
from typing import Dict, Tuple, Callable

import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path, help="path to training settings")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    settings = read_settings(args.settings)

    train, val, test, le, tokenizer = prepare_dataset(settings["dataset"], settings["model"], settings["num_workers"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Hello world!")


def read_settings(filename: Path) -> Dict:
    with open(filename, "r") as fp:
        return yaml.load(stream=fp, Loader=yaml.FullLoader)


def prepare_dataset(
        dataset_paths: Dict[str, str],
        model_params: Dict,
        num_workers: int = 1
) -> Tuple[Dataset, Dataset, Dataset, LabelEncoder, Callable]:
    train_df = pd.read_csv(dataset_paths["train"])
    val_df = pd.read_csv(dataset_paths["val"])
    test_df = pd.read_csv(dataset_paths["test"])

    le = LabelEncoder()
    train_df.rate = le.fit_transform(train_df.label)

    train = Dataset.from_pandas(train_df)
    val = Dataset.from_pandas(val_df)
    test = Dataset.from_pandas(test_df)

    # Text preprocessing.
    tokenizer = AutoTokenizer.from_pretrained(model_params["name"], max_len=model_params["max_len"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

    tokenized_train = train.map(tokenize_function, batched=True, num_proc=num_workers)
    tokenized_val = val.map(tokenize_function, batched=True, num_proc=num_workers)
    tokenized_test = test.map(tokenize_function, batched=True, num_proc=num_workers)

    return tokenized_train, tokenized_val, tokenized_test, le, tokenize_function


if __name__ == "__main__":
    main()
