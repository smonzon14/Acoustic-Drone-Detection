from __future__ import annotations

import argparse
import os
import random
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import DADSParquetDataset
from model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_spectrogram_batch(
    batch: Iterable[Tuple[torch.Tensor, int]],
    max_frames: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    specs = []
    labels = []

    for item in batch:
        if len(item) == 3:
            spec, label, _sample_rate = item
        else:
            spec, label = item

        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        elif spec.dim() == 3:
            pass
        else:
            raise ValueError(
                "Expected spectrogram with shape (F, T) or (C, F, T). "
                "Enable to_spectrogram in the dataset."
            )

        if max_frames is not None and spec.shape[-1] > max_frames:
            spec = spec[..., :max_frames]
        specs.append(spec)
        labels.append(int(label))

    max_freq = max(spec.shape[-2] for spec in specs)
    max_time = max(spec.shape[-1] for spec in specs)
    if max_frames is not None:
        max_time = min(max_time, max_frames)

    padded = []
    for spec in specs:
        pad_freq = max_freq - spec.shape[-2]
        if max_frames is not None and spec.shape[-1] > max_time:
            spec = spec[..., :max_time]
        pad_time = max_time - spec.shape[-1]
        if pad_freq or pad_time:
            spec = F.pad(spec, (0, pad_time, 0, pad_freq))
        padded.append(spec)

    batch_specs = torch.stack(padded, dim=0)
    batch_labels = torch.tensor(labels, dtype=torch.float32)
    return batch_specs, batch_labels


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    for step, (inputs, labels) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs).view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.numel()

        if log_interval and step % log_interval == 0:
            avg_loss = total_loss / max(1, total_samples)
            avg_acc = total_correct / max(1, total_samples)
            print(f"step {step}: loss={avg_loss:.4f} acc={avg_acc:.4f}")

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs).view(-1)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.numel()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drone/no-drone CNN.")
    parser.add_argument("--data-dir", default="detection/local_dataset_dir/data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--save-path", default="checkpoints/cnn.pt")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = DADSParquetDataset(data_dir=args.data_dir, to_spectrogram=True)
    if args.val_split > 0.0:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        train_set = dataset
        val_set = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_spectrogram_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_spectrogram_batch,
            pin_memory=device.type == "cuda",
        )

    model = build_model(num_classes=1, in_channels=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    last_improve_step = 0
    patience_steps = 50
    global_step = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for _step, (inputs, labels) in enumerate(train_loader, start=1):
            global_step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if args.log_interval and global_step % args.log_interval == 0:
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                acc = (preds == labels).float().mean().item()
                print(f"step {global_step}: loss={loss.item():.4f} acc={acc:.4f}")

            if val_loader is not None and args.evaluation_interval > 0:
                if global_step % args.evaluation_interval == 0:
                    val_loss, val_acc = evaluate(model, val_loader, device)
                    print(
                        f"step {global_step}: val_loss={val_loss:.4f} "
                        f"val_acc={val_acc:.4f}"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        last_improve_step = global_step
                        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
                        torch.save({"model_state": model.state_dict()}, args.save_path)
                        print(f"saved model to {args.save_path}")
                    elif global_step - last_improve_step >= patience_steps:
                        print(
                            "early stopping: no val_loss improvement for "
                            f"{patience_steps} steps"
                        )
                        return

    if val_loader is None:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({"model_state": model.state_dict()}, args.save_path)
        print(f"saved model to {args.save_path}")


if __name__ == "__main__":
    main()
