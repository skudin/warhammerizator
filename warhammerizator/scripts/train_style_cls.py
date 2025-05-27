import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple, Callable

import yaml
import torch
import evaluate
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, PreTrainedTokenizerBase

from warhammerizator import conf


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path, help="path to training settings")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    settings = read_settings(args.settings)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train, val, test, le, tokenizer = prepare_dataset(settings["dataset"], settings["model"], settings["num_workers"])

    train_cls(settings["experiment"], settings["model"], settings["training"], train, val, tokenizer, device)

    print("Hello world!")


def read_settings(filename: Path) -> Dict:
    with open(filename, "r") as fp:
        return yaml.load(stream=fp, Loader=yaml.FullLoader)


def prepare_dataset(
        dataset_paths: Dict[str, str],
        model_params: Dict,
        num_workers: int = 1
) -> Tuple[Dataset, Dataset, Dataset, LabelEncoder, PreTrainedTokenizerBase]:
    train_df = pd.read_csv(dataset_paths["train"])[: 1000]
    val_df = pd.read_csv(dataset_paths["val"])[: 1000]
    test_df = pd.read_csv(dataset_paths["test"])[: 1000]

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

    return tokenized_train, tokenized_val, tokenized_test, le, tokenizer


def train_cls(
        experiment: str,
        model_params: Dict,
        training_params: Dict,
        train: Dataset,
        val: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        device: str
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_params["name"],
        num_labels=model_params["num_labels"],
        device_map=device,
        classifier_dropout=model_params["classifier_dropout"]
    )

    training_args = TrainingArguments(output_dir=conf.ROOT_PATH / "experiments" / experiment, **training_params)

    f1_score = evaluate.load("f1")
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "f1": f1_score.compute(predictions=predictions, references=labels, average="macro")["f1"],
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    with mlflow.start_run():
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        trainer.train()

        # Save model.
        save_directory = conf.ROOT_PATH / "data" / "models" / experiment
        if save_directory.exists():
            shutil.rmtree(save_directory)
        save_directory.mkdir(parents=True)

        tokenizer.save_pretrained(save_directory)
        model.save_pretrained(save_directory)

    pass


if __name__ == "__main__":
    main()
