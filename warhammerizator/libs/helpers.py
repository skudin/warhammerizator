import shutil
import random
from pathlib import Path

import numpy as np
import torch

from warhammerizator import conf


def create_output_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)


def create_folder_with_dialog(path: Path) -> None:
    if path.exists():
        answer = "#"
        while answer not in "yYnN":
            answer = input(f"Remove {path.as_posix()} (y/n)\n")
            if answer in "yY":
                shutil.rmtree(path)
            else:
                exit(0)

    path.mkdir(parents=True)


def seed_everything() -> None:
    random.seed(conf.RANDOM_STATE)
    np.random.seed(conf.RANDOM_STATE)
    torch.manual_seed(conf.RANDOM_STATE)
    torch.cuda.manual_seed_all(conf.RANDOM_STATE)
