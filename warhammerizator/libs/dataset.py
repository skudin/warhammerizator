from typing import List

from torch.utils.data import Dataset


class MonostyleDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            style: str,
            part: str
    ):
        super().__init__()

        self._dataset_path = dataset_path
        self._style = style
        self._part = part

        self._data = self._load_data()

    def _load_data(self) -> List[str]:
        with open(self._dataset_path) as fp:
            data = [line.strip() for line in fp][:128]

        print(f"Dataset {self._style}, part {self._part}: {len(data)} examples.")

        return data

    def reduce_data(self, n_samples):
        self._data = self._data[:n_samples]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx) -> str:
        return self._data[idx]
