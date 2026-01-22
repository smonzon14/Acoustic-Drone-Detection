from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

import torch

from dataset import DADSParquetDataset
from model import build_model


def save_wav(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    if waveform.dim() != 1:
        waveform = waveform.view(-1)

    try:
        import soundfile as sf  # type: ignore

        sf.write(path, waveform.cpu().numpy(), sample_rate)
        return
    except Exception:
        pass

    try:
        import numpy as np  # type: ignore
        from scipy.io import wavfile  # type: ignore

        audio = waveform.cpu().numpy()
        audio = audio.astype("float32")
        audio = np.clip(audio, -1.0, 1.0)
        wavfile.write(path, sample_rate, (audio * 32767).astype("int16"))
    except Exception as exc:
        raise RuntimeError(
            "Unable to save wav file. Install `soundfile` or `scipy`."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved CNN on full dataset.")
    parser.add_argument("--data-dir", default="detection/local_dataset_dir/data")
    parser.add_argument("--checkpoint", default="checkpoints/cnn.pt")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-dir", default="checkpoints/worst_samples")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = DADSParquetDataset(
        data_dir=args.data_dir, to_spectrogram=False, return_sample_rate=True
    )

    model = build_model(num_classes=1, in_channels=1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    worst: List[Tuple[float, int, float, int, str]] = []

    os.makedirs(args.output_dir, exist_ok=True)
    worst_csv = os.path.join(args.output_dir, "worst_scores.csv")

    with torch.no_grad():
        for idx in range(len(dataset)):
            waveform, label, sample_rate = dataset[idx]
            sample_rate = sample_rate or 16000
            spec = dataset._to_spectrogram(waveform)

            if args.max_frames is not None and spec.shape[-1] > args.max_frames:
                spec = spec[..., : args.max_frames]
            if args.min_frames is not None and spec.shape[-1] < args.min_frames:
                pad_time = args.min_frames - spec.shape[-1]
                spec = torch.nn.functional.pad(spec, (0, pad_time))

            spec = spec.unsqueeze(0).unsqueeze(0).to(device)
            logits = model(spec).view(-1)
            prob = torch.sigmoid(logits)[0].item()
            error = abs(prob - float(label))

            worst.append((error, idx, prob, int(label), ""))
            worst.sort(key=lambda x: x[0], reverse=True)
            if len(worst) > args.top_k:
                worst.pop()

    with open(worst_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "prob", "error", "wav_path"])

        for rank, (error, idx, prob, label, _path) in enumerate(worst, start=1):
            waveform, _, sample_rate = dataset[idx]
            sample_rate = sample_rate or 16000
            wav_path = os.path.join(args.output_dir, f"worst_{rank:02d}_idx_{idx}.wav")
            save_wav(wav_path, waveform, sample_rate)
            writer.writerow([idx, label, f"{prob:.6f}", f"{error:.6f}", wav_path])

    print(f"Saved top {args.top_k} worst samples to {worst_csv}")


if __name__ == "__main__":
    main()
