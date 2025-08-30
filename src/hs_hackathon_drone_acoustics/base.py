from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import overload

import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

from hs_hackathon_drone_acoustics import CLASSES


@dataclass(frozen=True)
class AudioWaveform:
    data: Tensor
    sample_rate: float

    @classmethod
    def load(cls, path: Path) -> AudioWaveform:
        data, samplerate = sf.read(path)
        return AudioWaveform(torch.as_tensor(data, dtype=torch.float32), samplerate)

    @property
    def duration(self) -> float:
        return self.data.shape[-1] / self.sample_rate


class AudioDataset(Dataset[tuple[AudioWaveform, int]]):
    def __init__(self, root_dir: Path) -> None:
        if not root_dir.is_dir():
            raise FileNotFoundError(f"{root_dir} is not a directory")
        self._root_dir = root_dir
        self._filepaths: list[Path] = []
        self._labels: list[int] = []
        self._class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        # Parse labels and filenames from class directories
        for clz in CLASSES:
            class_dir = root_dir / clz
            for file_name in class_dir.glob("*.wav"):
                self._filepaths.append(class_dir / file_name)
                self._labels.append(self._class_to_idx[clz])

    def __len__(self) -> int:
        return len(self._filepaths)

    @overload
    def __getitem__(self, idx: int) -> tuple[AudioWaveform, int]: ...

    @overload
    def __getitem__(self, idx: slice) -> tuple[list[AudioWaveform], list[int]]: ...

    def __getitem__(self, idx: int | slice) -> tuple[AudioWaveform, int] | tuple[list[AudioWaveform], list[int]]:
        if isinstance(idx, int):
            waveform = AudioWaveform.load(self._filepaths[idx])
            label = self._labels[idx]
            return waveform, label
        else:
            list_waveforms = []
            labels = []
            for sub_idx in range(len(self))[idx]:
                single_waveform, single_label = self.__getitem__(sub_idx)
                list_waveforms.append(single_waveform)
                labels.append(single_label)
            return list_waveforms, labels
