from pathlib import Path
import pickle

import torch
import torchaudio
import numpy as np


class MusicGenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = None,
        np_list: np.ndarray = None,
        np_addr_list: np.ndarray = None,
        target_duration: float = 1.0,
        target_sample_rate: int = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.__check([self.root])
        self.np_list = np_list
        self.np_addr_list = np_addr_list
        self.target_duration = target_duration
        self.target_sample_rate = target_sample_rate
        assert self.target_sample_rate is not None

    def __check(self, path_list: list) -> None:
        for path in path_list:
            path: Path
            assert path.exists(), f"{path.as_posix()} NOT FOUND"

    def __load_audio(self, path: Path) -> tuple:
        data, sr = torchaudio.load(path)
        duration_in_seconds = data.size(-1) // sr
        return data, sr, duration_in_seconds

    def __uniform_random(self, start: float, end: float) -> float:
        result = torch.rand(1) * (end - start) + start
        return result.item()

    def __db2linear(self, db: float) -> float:
        import math

        return math.pow(10, db / 10)

    def __len__(self):
        return len(self.np_addr_list)

    def __getitem__(self, idx):
        start = 0 if idx == 0 else self.np_addr_list[idx - 1]
        end = self.np_addr_list[idx]
        file_path = pickle.loads(memoryview(self.np_list[start:end]))
        path = self.root.joinpath(file_path)
        data, sr, duration_in_seconds = self.__load_audio(path)
        if duration_in_seconds < self.target_duration:
            return None
        target_sr_len = int(self.target_duration * self.target_sample_rate)
        src_sr_len = int(self.target_duration * sr)
        data_len = data.size(-1)
        # sample data
        start_idx = 0
        if data_len > src_sr_len:
            start_idx = torch.randint(
                low=0, high=data_len - src_sr_len, size=(1,)
            ).item()
        samples = data[..., start_idx : start_idx + src_sr_len]
        assert (
            samples.size(-1) == src_sr_len
        ), f"size mismatches {samples.size()} vs {src_sr_len=} at {path=}"
        # stereo to mono
        if samples.size(0) > 1:
            samples = samples.mean(0, keepdim=True)
        if sr != self.target_sample_rate:
            samples = torchaudio.functional.resample(
                samples, sr, self.target_sample_rate
            )
            assert (
                samples.size(-1) == target_sr_len
            ), f"size mimatches after resampling {samples.size()} vs {target_sr_len} at {path=}"
        return samples
