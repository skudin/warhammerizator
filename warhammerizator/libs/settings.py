from pathlib import Path
from typing import Dict

import yaml


class Settings:
    def __init__(self, filename: Path):
        file_content = self._read_settings_file(filename)

        self._init_fields(file_content)
        self._postprocessing()

    def _read_settings_file(self, filename: Path):
        with open(filename, "r") as fp:
            return yaml.load(stream=fp, Loader=yaml.FullLoader)

    def _init_fields(self, file_content: Dict) -> None:
        # Hyper params.
        # Optimizer.
        self.optimizer = file_content["hparams"]["optimizer"]

        # Scheduler.
        self.scheduler = file_content["hparams"]["scheduler"]

        # Loss.
        self.loss = file_content["hparams"]["loss"]

        # Models.
        self.models = file_content["hparams"]["models"]

        # Trainer.
        self.trainer = file_content["hparams"]["trainer"]

        # Data.
        self.dataset = file_content["data"]["dataset"]
        self.data_loader = file_content["data"]["data_loader"]

        # Logger.
        self.logger = file_content["logger"]

        # Common.
        self.common = file_content["common"]

    def _postprocessing(self) -> None:
        self.trainer["save_base_folder"] = Path(self.trainer["save_base_folder"])
        self.trainer["save_base_folder"] /= self.common["experiment"]
        if self.trainer["from_pretrained"] is not None:
            self.trainer["from_pretrained"] = Path(self.trainer["from_pretrained"])

        self.models["pretrained_classifier_model"] = Path(self.models["pretrained_classifier_model"])

        self.dataset["wh40k_dataset"] = Path(self.dataset["wh40k_dataset"])
        self.dataset["normal_style_dataset"] = Path(self.dataset["normal_style_dataset"])
