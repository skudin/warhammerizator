import copy
import json
from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pprint
from typing import Iterable, Dict

import numpy as np
from razdel import sentenize

from warhammerizator.libs import helpers


class EDABase(ABC):
    def __init__(
            self,
            input_path: Path,
            stats_parts: Iterable[str],
            dataset_parts: Dict[str, str]
    ):
        self._input_path = input_path
        self._stats_parts = stats_parts
        self._dataset_parts = dataset_parts

    def processing(self) -> Dict[str, Dict[str, Dict[str, int | Dict[str, int | float]]]]:
        stats = {part: dict() for part in self._dataset_parts.keys()}

        for dataset_part, filename in self._dataset_parts.items():
            dataset_part_stats = self._processing_dataset_part(self._input_path / filename)

            for stats_part, item in dataset_part_stats.items():
                stats[dataset_part][stats_part] = item

        return stats

    def save_stats(
            self,
            stats: Dict[str, Dict[str, Dict[str, int | Dict[str, int | float]]]],
            output_path: Path
    ) -> None:
        helpers.create_folder_with_dialog(output_path)

        for dataset_part, dataset_part_stats in stats.items():
            with open(output_path / f"{dataset_part}.json", "w") as fp:
                json.dump(obj=dataset_part_stats, fp=fp, indent=4, sort_keys=False)

    def print_stats(self, stats: Dict[str, Dict[str, Dict[str, int | Dict[str, int | float]]]]) -> None:
        for dataset_part, dataset_part_stats in stats.items():
            print(f"{dataset_part} stats:")
            pprint(dataset_part_stats, sort_dicts=False)
            print()

    def _processing_dataset_part(self, filename: Path) -> Dict[str, Dict[str, int | Dict[str, int | float]]]:
        template = {
            "samples": 0,
            "num_sentences": list(),
            "num_words": list(),
            "num_characters": list()
        }
        tmp = {stats_part: copy.deepcopy(template) for stats_part in self._stats_parts}

        with open(filename, "r") as fp:
            for line in fp:
                data = self._processing_line(line)

                for stats_part, text in data.items():
                    tmp[stats_part]["num_sentences"].append(len(list(sentenize(text))))
                    tmp[stats_part]["num_words"].append(len(text.split(" ")))
                    tmp[stats_part]["num_characters"].append(len(text))
                    tmp[stats_part]["samples"] += 1

        stats = dict()
        for stats_part, dataset_part_stats in tmp.items():
            stats[stats_part] = {
                "samples": dataset_part_stats["samples"],
                "sentence": {
                    "mean": float(np.mean(dataset_part_stats["num_sentences"])),
                    "std": float(np.std(dataset_part_stats["num_sentences"])),
                    "min": int(np.min(dataset_part_stats["num_sentences"])),
                    "max": int(np.max(dataset_part_stats["num_sentences"]))
                },
                "word": {
                    "mean": float(np.mean(dataset_part_stats["num_words"])),
                    "std": float(np.std(dataset_part_stats["num_words"])),
                    "min": int(np.min(dataset_part_stats["num_words"])),
                    "max": int(np.max(dataset_part_stats["num_words"]))
                },
                "character": {
                    "mean": float(np.mean(dataset_part_stats["num_characters"])),
                    "std": float(np.std(dataset_part_stats["num_characters"])),
                    "min": int(np.min(dataset_part_stats["num_characters"])),
                    "max": int(np.max(dataset_part_stats["num_characters"]))
                }
            }

        return stats

    @abstractmethod
    def _processing_line(self, line: str) -> Dict[str, str]:
        pass
