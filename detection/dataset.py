from __future__ import annotations

import bisect
import glob
import io
import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


class DADSParquetDataset(Dataset):
    """PyTorch dataset that streams audio + label pairs from local parquet shards."""

    def __init__(
        self,
        data_dir: str = "detection/local_dataset_dir/data",
        audio_column: str = "audio",
        label_column: str = "label",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_sample_rate: bool = False,
        to_spectrogram: bool = True,
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        power: float = 2.0,
    ) -> None:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required to read parquet files. Install with `pip install pyarrow`."
            ) from exc

        self._pq = pq
        self.data_dir = data_dir
        self.audio_column = audio_column
        self.label_column = label_column
        self.transform = transform
        self.return_sample_rate = return_sample_rate
        self.to_spectrogram = to_spectrogram
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power

        self._files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not self._files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        self._row_groups = []
        self._row_group_starts = []
        self._total_rows = 0
        for path in self._files:
            parquet_file = pq.ParquetFile(path)
            for rg_index in range(parquet_file.num_row_groups):
                rg_rows = parquet_file.metadata.row_group(rg_index).num_rows
                self._row_group_starts.append(self._total_rows)
                self._row_groups.append((path, rg_index, rg_rows))
                self._total_rows += rg_rows

        self._parquet_files = {}

    def __len__(self) -> int:
        return self._total_rows

    def __getitem__(self, index: int):
        if index < 0:
            index += self._total_rows
        if index < 0 or index >= self._total_rows:
            raise IndexError(f"Index {index} out of range for dataset of size {self._total_rows}")

        rg_pos = bisect.bisect_right(self._row_group_starts, index) - 1
        path, rg_index, _ = self._row_groups[rg_pos]
        row_in_group = index - self._row_group_starts[rg_pos]

        parquet_file = self._get_parquet_file(path)
        table = parquet_file.read_row_group(
            rg_index, columns=[self.audio_column, self.label_column]
        )
        row = table.slice(row_in_group, 1).to_pydict()
        audio = row[self.audio_column][0]
        label = row[self.label_column][0]

        waveform, sample_rate = self._normalize_audio(audio)
        label = self._normalize_label(label)

        if self.to_spectrogram:
            waveform = self._to_spectrogram(waveform)

        if self.transform is not None:
            waveform = self.transform(waveform)

        if self.return_sample_rate:
            return waveform, label, sample_rate
        return waveform, label

    def _get_parquet_file(self, path: str):
        parquet_file = self._parquet_files.get(path)
        if parquet_file is None:
            parquet_file = self._pq.ParquetFile(path)
            self._parquet_files[path] = parquet_file
        return parquet_file

    def _normalize_audio(self, audio) -> Tuple[torch.Tensor, Optional[int]]:
        if isinstance(audio, dict):
            if "array" in audio:
                waveform = torch.tensor(audio["array"], dtype=torch.float32)
                return waveform, audio.get("sampling_rate")
            if "bytes" in audio:
                return self._decode_audio_bytes(audio["bytes"])
            if "path" in audio and audio["path"]:
                return self._decode_audio_path(audio["path"])

        if isinstance(audio, (bytes, bytearray)):
            return self._decode_audio_bytes(audio)

        if isinstance(audio, (list, tuple)):
            return torch.tensor(audio, dtype=torch.float32), None

        if hasattr(audio, "shape"):
            return torch.as_tensor(audio, dtype=torch.float32), None

        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _decode_audio_bytes(self, data: bytes) -> Tuple[torch.Tensor, Optional[int]]:
        try:
            import soundfile as sf  # type: ignore

            array, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            return torch.tensor(array, dtype=torch.float32), int(sample_rate)
        except Exception:
            pass

        try:
            import numpy as np  # type: ignore
            from scipy.io import wavfile  # type: ignore

            sample_rate, array = wavfile.read(io.BytesIO(data))
            if array.ndim > 1:
                array = array.mean(axis=1)
            if array.dtype.kind in ("i", "u"):
                max_val = float(np.iinfo(array.dtype).max)
                if max_val:
                    array = array.astype("float32") / max_val
            return torch.tensor(array, dtype=torch.float32), int(sample_rate)
        except Exception as exc:
            raise RuntimeError(
                "Unable to decode audio bytes. Install `soundfile` or `scipy`."
            ) from exc

    def _decode_audio_path(self, path: str) -> Tuple[torch.Tensor, Optional[int]]:
        try:
            import soundfile as sf  # type: ignore

            array, sample_rate = sf.read(path, dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            return torch.tensor(array, dtype=torch.float32), int(sample_rate)
        except Exception:
            pass

        try:
            import numpy as np  # type: ignore
            from scipy.io import wavfile  # type: ignore

            sample_rate, array = wavfile.read(path)
            if array.ndim > 1:
                array = array.mean(axis=1)
            if array.dtype.kind in ("i", "u"):
                max_val = float(np.iinfo(array.dtype).max)
                if max_val:
                    array = array.astype("float32") / max_val
            return torch.tensor(array, dtype=torch.float32), int(sample_rate)
        except Exception as exc:
            raise RuntimeError(
                "Unable to decode audio path. Install `soundfile` or `scipy`."
            ) from exc

    @staticmethod
    def _normalize_label(label) -> int:
        if isinstance(label, dict) and "label" in label:
            return int(label["label"])
        return int(label)

    def _to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 1:
            waveform = waveform.view(-1)

        hop_length = self.hop_length if self.hop_length is not None else self.n_fft // 4
        win_length = self.win_length if self.win_length is not None else self.n_fft
        window = torch.hann_window(win_length, device=waveform.device)
        if waveform.numel() < self.n_fft:
            pad_amount = self.n_fft - waveform.numel()
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        spec = stft.abs()
        if self.power != 1.0:
            spec = spec.pow(self.power)
        return torch.log1p(spec)

if __name__ == "__main__":
    import random

    data_dir = "detection/local_dataset_dir/data"
    raw_dataset = DADSParquetDataset(data_dir=data_dir, to_spectrogram=False)
    spec_dataset = DADSParquetDataset(data_dir=data_dir, to_spectrogram=True)

    print(f"Dataset size: {len(raw_dataset)}")

    sample_count = 20
    random.seed(42)
    indices = [random.randrange(len(raw_dataset)) for _ in range(sample_count)]

    raw_lengths = []
    spec_shapes = []

    for idx in indices:
        waveform, label = raw_dataset[idx]
        spec, _ = spec_dataset[idx]
        raw_lengths.append(int(waveform.numel()))
        spec_shapes.append(tuple(spec.shape))

    print(f"Sampled {sample_count} items")
    print(f"Raw waveform length (samples): min={min(raw_lengths)} max={max(raw_lengths)}")
    print(f"Spectrogram shapes (F x T): min={min(spec_shapes)} max={max(spec_shapes)}")
