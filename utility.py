import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset  # type: ignore
from PIL import Image  # type: ignore
from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset_builder.core.utility import load_manifest_parquet
from sklearn.metrics import classification_report


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.image_labels = load_manifest_parquet(data_path)[:10]
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        img_path, label = self.image_labels[index]
        # img_full_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def model_builder(num_species, is_eval=False):
    if is_eval:
        model = models.mobilenet_v3_large(weights=None)
    else:
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )

    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_species)
    return model


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", end=" | ")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("No GPU found")
    return device


def build_dataloaders(
    data_dir: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=32.0 / 255.0, saturation=0.5, contrast=0.5, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    transform_val = transforms.Compose(
        [
            # Resize to maintain 87.5% central fraction
            transforms.Resize(int(224 / 0.875)),
            transforms.CenterCrop(224),  # Central cropping (87.5%)
            transforms.ToTensor(),  # Convert image to tensor [0, 1]
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # Normalize to [-1, 1]
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    train_dataset = CustomDataset(
        os.path.join(data_dir, "train.parquet"), transform_train
    )
    val_dataset = CustomDataset(os.path.join(data_dir, "val.parquet"), transform_val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
):
    model.train()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training", unit="batch")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Tensor guard
        if labels.min() < 0 or labels.max() >= outputs.shape[1]:
            print("Invalid labels detected!")
            print(f"Labels: {labels}")
            print(f"Label min: {labels.min().item()} | max: {labels.max().item()}")
            print(f"Number of classes (output.shape[1]): {outputs.shape[1]}")
            raise ValueError("Label out of range for CrossEntropyLoss.")

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(loss=f"{loss.detach().item():.3f}")

    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    return avg_loss, accuracy


def train_one_epoch_amp(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
):
    model.train()
    total_loss, correct = 0.0, 0
    loop = tqdm(dataloader, desc="Training")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)

            # Tensor guard
            if labels.min() < 0 or labels.max() >= outputs.shape[1]:
                print("Invalid labels detected!")
                print(f"Labels: {labels}")
                print(f"Label min: {labels.min().item()} | max: {labels.max().item()}")
                print(f"Number of classes (output.shape[1]): {outputs.shape[1]}")
                raise ValueError("Label out of range for CrossEntropyLoss.")

            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        loop.set_postfix(loss=f"{loss.item():.3f}")
    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    return avg_loss, accuracy


def validate(
    model: torch.nn.Module, dataloader: DataLoader, criterion, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)  # type: ignore
    accuracy = correct / len(dataloader.dataset)  # type: ignore
    return avg_loss, accuracy


def save_model(model: torch.nn.Module, name: str, device: torch.device):
    os.makedirs("./models/new_model", exist_ok=True)
    model_path = f"./models/new_model/{name}.pth"
    torch.save(model, model_path)
    print(f"Saved model to {model_path}")

    # ONNX export
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    export_path = f"./models/new_model/{name}.onnx"
    to_export = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.onnx.export(
        to_export,
        dummy_input,  # type: ignore
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {export_path}")


def get_support_list(json_path: str, species_name: List[str]) -> List[int]:
    with open(json_path, "r") as f:
        species_count_dict = json.load(f)
    total_support_list = [species_count_dict.get(name, 0) for name in species_name]
    return total_support_list


def generate_report(
    all_labels: List[np.ndarray],
    all_preds: List[np.ndarray],
    species_names: List[str],
    total_support_list: List[int],
    accuracy: float,
) -> pd.DataFrame:
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=species_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    val_support_series = report_df.loc[species_names, "support"]
    total_support_series = pd.Series(total_support_list, index=species_names)
    train_support_series = total_support_list - val_support_series
    report_df.loc[species_names, "train_support"] = train_support_series.astype(int)
    report_df.loc[species_names, "total_support"] = total_support_series.astype(int)
    report_df.loc[species_names, "support"] = report_df.loc[
        species_names, "support"
    ].astype(int)

    species_df = report_df.loc[species_names].copy()
    summary_df = report_df.drop(index=species_names).copy()
    species_df = species_df.sort_values(by="f1-score", ascending=True)
    report_df = pd.concat([species_df, summary_df])
    report_df.loc["accuracy"] = {
        "precision": accuracy,
        "recall": 0,
        "f1-score": 0,
        "support": np.nan,
    }
    report_df.loc["macro avg", "support"] = np.nan
    report_df.loc["weighted avg", "support"] = np.nan
    return report_df
