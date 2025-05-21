import argparse
import os
import pickle
from pathlib import Path
from typing import Tuple

import torch
import mlflow
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

from warhammerizator.libs.settings import Settings
from warhammerizator.libs.dataset import MonostyleDataset
from warhammerizator.libs.generator_model import GeneratorModel
from warhammerizator.libs.discriminator_model import DiscriminatorModel
from warhammerizator.libs.classifier_model import ClassifierModel
from warhammerizator.libs.cycle_gan_model import CycleGANModel
from warhammerizator.libs.evaluator import Evaluator


def parse_command_prompt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", type=Path, help="path to file with training settings")

    return parser.parse_args()


def main():
    args = parse_command_prompt()

    settings = Settings(args.settings)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    mlflow.set_tracking_uri(settings.logger["tracking_uri"])
    mlflow.set_experiment(settings.common["experiment"])

    mono_ds_a_train, mono_ds_b_train, mono_ds_a_eval, mono_ds_b_eval = init_datasets(settings)
    mono_dl_a_train, mono_dl_b_train, mono_dl_b_eval, mono_dl_a_eval = init_data_loaders(
        settings, mono_ds_a_train, mono_ds_b_train, mono_ds_a_eval, mono_ds_b_eval)

    G_ab, G_ba = init_generators(settings)
    D_ab, D_ba = init_discriminators(settings)
    Cls = init_classifier(settings)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cycleGAN = CycleGANModel(G_ab, G_ba, D_ab, D_ba, Cls, device=device)

    n_batch_epoch = min(len(mono_dl_a_train), len(mono_dl_b_train))
    num_training_steps = settings.trainer["epochs"] * n_batch_epoch

    print(f"Total number of training steps: {num_training_steps}")

    warmup_steps = int(0.1 * num_training_steps) if settings.scheduler["warmup"] else 0

    optimizer = AdamW(cycleGAN.get_optimizer_parameters(), lr=settings.optimizer["learning_rate"])
    lr_scheduler = get_scheduler(settings.scheduler["type"], optimizer=optimizer, num_warmup_steps=warmup_steps,
                                 num_training_steps=num_training_steps)

    start_epoch = 0
    current_training_step = 0

    if settings.trainer["from_pretrained"] is not None:
        checkpoint = torch.load(settings.trainer["from_pretrained"] / "checkpoint.pth", map_location=torch.device("cpu"))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        current_training_step = checkpoint['training_step']
        del checkpoint

    loss_logging = {'Cycle Loss A-B-A': [], 'Loss generator  A-B': [], 'Classifier-guided A-B': [], 'Loss D(A->B)': [],
                    'Cycle Loss B-A-B': [], 'Loss generator  B-A': [], 'Classifier-guided B-A': [], 'Loss D(B->A)': []}

    progress_bar = tqdm(range(num_training_steps))
    progress_bar.update(current_training_step)

    print('Start training...')
    with mlflow.start_run() as mlflow_run:
        evaluator = Evaluator(cycleGAN, settings, mlflow_run)
        for epoch in range(start_epoch, settings.trainer["epochs"]):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                print(f"\nTraining epoch: {epoch}")
                cycleGAN.train()  # set training mode

                for unsupervised_a, unsupervised_b in zip(mono_dl_a_train, mono_dl_b_train):
                    len_a, len_b = len(unsupervised_a), len(unsupervised_b)
                    if len_a > len_b:
                        unsupervised_a = unsupervised_a[:len_b]
                    else:
                        unsupervised_b = unsupervised_b[:len_a]

                    cycleGAN.training_cycle(sentences_a=unsupervised_a,
                                            sentences_b=unsupervised_b,
                                            lambdas=settings.loss["lambdas"],
                                            mlflow_run=mlflow_run,
                                            loss_logging=loss_logging,
                                            training_step=current_training_step)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    current_training_step += 1

                evaluator.run_eval_mono(epoch, current_training_step, 'validation', mono_dl_a_eval, mono_dl_b_eval)

                if epoch % settings.trainer["save_steps"] == 0:
                    cycleGAN.save_models(settings.trainer["save_base_folder"] / f"epoch_{epoch}")
                    checkpoint = {'epoch': epoch + 1, 'training_step': current_training_step,
                                  'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}
                    torch.save(checkpoint, settings.trainer["save_base_folder"] / f"epoch_{epoch}" / "checkpoint.pth")
                    if epoch > 0 and os.path.exists(settings.trainer["save_base_folder"] / f"epoch_{epoch - 1}" / "checkpoint.pth"):
                        os.remove(settings.trainer["save_base_folder"] / f"epoch_{epoch - 1}" / "checkpoint.pth")
                    if epoch > 0 and os.path.exists(settings.trainer["save_base_folder"] / "loss.pickle"):
                        os.remove(settings.trainer["save_base_folder"] / "loss.pickle")

                        with open(settings.trainer["save_base_folder"] / "loss.pickle", 'wb') as fp:
                            pickle.dump(loss_logging, fp)

                if settings.trainer["control_file"] is not None and os.path.exists(settings.trainer["control_file"]):
                    with open(args.control_file, 'r') as f:
                        if f.read() == 'STOP':
                            print(f'STOP command received - Stopped at epoch {epoch}')
                            os.remove(args.control_file)
                            break
                cycleGAN.train()

        print('End training...')

    print("Done.")


def init_datasets(settings: Settings) -> Tuple[MonostyleDataset, MonostyleDataset, MonostyleDataset, MonostyleDataset]:
    mono_ds_a_train = MonostyleDataset(
        dataset_path=settings.dataset["normal_style_dataset"] / "train.txt",
        style="normal",
        part="train"
    )
    mono_ds_b_train = MonostyleDataset(
        dataset_path=settings.dataset["wh40k_dataset"] / "train.txt",
        style="wh40k",
        part="train"
    )

    mono_ds_a_eval = MonostyleDataset(
        dataset_path=settings.dataset["normal_style_dataset"] / "val.txt",
        style="normal",
        part="eval"
    )
    mono_ds_b_eval = MonostyleDataset(
        dataset_path=settings.dataset["wh40k_dataset"] / "val.txt",
        style="wh40k",
        part="eval"
    )

    return mono_ds_a_train, mono_ds_b_train, mono_ds_a_eval, mono_ds_b_eval


def init_data_loaders(
        settings: Settings,
        mono_ds_a_train: MonostyleDataset,
        mono_ds_b_train: MonostyleDataset,
        mono_ds_a_eval: MonostyleDataset,
        mono_ds_b_eval: MonostyleDataset
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    mono_dl_a_train = DataLoader(
        mono_ds_a_train,
        batch_size=settings.data_loader["batch_size"],
        shuffle=settings.data_loader["shuffle"],
        num_workers=settings.data_loader["num_workers"],
        pin_memory=settings.data_loader["pin_memory"]
    )

    mono_dl_b_train = DataLoader(
        mono_ds_b_train,
        batch_size=settings.data_loader["batch_size"],
        shuffle=settings.data_loader["shuffle"],
        num_workers=settings.data_loader["num_workers"],
        pin_memory=settings.data_loader["pin_memory"]
    )

    mono_dl_a_eval = DataLoader(
        mono_ds_a_eval,
        batch_size=settings.data_loader["batch_size"],
        shuffle=settings.data_loader["shuffle"],
        num_workers=settings.data_loader["num_workers"],
        pin_memory=settings.data_loader["pin_memory"]
    )

    mono_dl_b_eval = DataLoader(
        mono_ds_b_eval,
        batch_size=settings.data_loader["batch_size"],
        shuffle=settings.data_loader["shuffle"],
        num_workers=settings.data_loader["num_workers"],
        pin_memory=settings.data_loader["pin_memory"]
    )

    return mono_dl_a_train, mono_dl_b_train, mono_dl_b_eval, mono_dl_a_eval


def init_generators(settings: Settings) -> Tuple[GeneratorModel, GeneratorModel]:
    if settings.trainer["from_pretrained"] is not None:
        G_ab = GeneratorModel(
            settings.models["generator_model_tag"],
            settings.trainer["from_pretrained"] / "G_ab",
            max_seq_length=settings.dataset["max_sequence_length"]
        )
        G_ba = GeneratorModel(
            settings.models["generator_model_tag"],
            settings.trainer["from_pretrained"] / "G_ba",
            max_seq_length=settings.dataset["max_sequence_length"]
        )
    else:
        G_ab = GeneratorModel(
            settings.models["generator_model_tag"],
            max_seq_length=settings.dataset["max_sequence_length"]
        )
        G_ba = GeneratorModel(
            settings.models["generator_model_tag"],
            max_seq_length=settings.dataset["max_sequence_length"]
        )

    return G_ab, G_ba


def init_discriminators(settings: Settings) -> Tuple[DiscriminatorModel, DiscriminatorModel]:
    if settings.trainer["from_pretrained"] is not None:
        D_ab = DiscriminatorModel(
            settings.models["discriminator_model_tag"],
            settings.trainer["from_pretrained"] / f"D_ab",
            max_seq_length=settings.dataset["max_sequence_length"]
        )
        D_ba = DiscriminatorModel(
            settings.models["discriminator_model_tag"],
            settings.trainer["from_pretrained"] / "D_ba",
            max_seq_length=settings.dataset["max_sequence_length"]
        )
    else:
        D_ab = DiscriminatorModel(
            settings.models["discriminator_model_tag"],
            max_seq_length=settings.dataset["max_sequence_length"]
        )
        D_ba = DiscriminatorModel(
            settings.models["discriminator_model_tag"],
            max_seq_length=settings.dataset["max_sequence_length"]
        )

    return D_ab, D_ba


def init_classifier(settings: Settings) -> ClassifierModel:
    Cls = ClassifierModel(
        settings.models["pretrained_classifier_model"],
        max_seq_length=settings.dataset["max_sequence_length"]
    )

    return Cls


if __name__ == "__main__":
    main()
