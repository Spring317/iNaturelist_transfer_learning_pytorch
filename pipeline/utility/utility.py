import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report
from dataset_builder import run_manifest_generator
from dataset_builder.core import load_config, validate_config
from dataset_builder.core.exceptions import ConfigError
from pipeline.dataset_loader import CustomDataset


def get_device(use_cpu=False) -> torch.device:
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"GPU model: {torch.cuda.get_device_name(device)}")
        else:
            print("No GPU found")
    print(f"Using device: {device}")
    return device


def mobile_net_v3_large_builder(
    device: torch.device,
    num_outputs: Optional[int] = None,
    start_with_weight=False,
    path: Optional[str] = None,
):
    if path and not num_outputs:
        # load full model
        model = torch.load(path, map_location=device, weights_only=False)
        model = model.to(device)

    else:
        if start_with_weight:
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            )
        else:
            model = models.mobilenet_v3_large(weights=None)
        old_linear_layer = model.classifier[3]
        assert isinstance(old_linear_layer, nn.Linear), "Expected a Linear layer"
        assert isinstance(num_outputs, int), (
            "Expected an int for classification layer output"
        )
        model.classifier[3] = nn.Linear(old_linear_layer.in_features, num_outputs)
        model = model.to(device)

    return model


def dataloader_wrapper(
    train_dataset: CustomDataset,
    val_dataset: CustomDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
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

        total_loss += loss.detach().item() * images.size(0)
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
    loop = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
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

        total_loss += loss.detach().item() * images.size(0)
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

    # extract support and compute training support
    val_support_series = report_df.loc[species_names, "support"].astype(int)
    total_support_series = pd.Series(total_support_list, index=species_names)
    train_support_series = total_support_list - val_support_series

    # add new column
    report_df.loc[species_names, "support"] = val_support_series
    report_df.loc[species_names, "train_support"] = train_support_series.astype(int)
    report_df.loc[species_names, "total_support"] = total_support_series.astype(int)

    # reorganize the report
    species_df = report_df.loc[species_names].copy()  # type: ignore
    summary_df = report_df.drop(index=species_names).copy()

    # sort species by f1-score (ascending)
    species_df = species_df.sort_values(by="f1-score", ascending=True)

    # combine details + summary
    report_df = pd.concat([species_df, summary_df])

    report_df.loc["accuracy"] = {  # type: ignore
        "precision": accuracy,
        "recall": 0,
        "f1-score": 0,
        "support": np.nan,
        "train_support": np.nan,
        "total_support": np.nan
    }

    # clean up summary support rows
    for row in ["macro avg", "weighted avg"]:
        for col in ["support", "train_support", "total_support"]:
            if col in report_df.columns:
                report_df.loc[row, col] = np.nan
    return report_df


def manifest_generator_wrapper(dominant_threshold: Optional[float] = None):
    try:
        config = load_config("./config.yaml")
        validate_config(config)
    except ConfigError as e:
        print(e)
        exit()

    target_classes = config["global"]["included_classes"]
    dst_dataset_path = config["paths"]["dst_dataset"]
    dst_dataset_name = os.path.basename(dst_dataset_path)
    output_path = config["paths"]["output_dir"]
    dst_properties_path = os.path.join(
        output_path, f"{dst_dataset_name}_composition.json"
    )
    train_size = config["train_val_split"]["train_size"]
    randomness = config["train_val_split"]["random_state"]
    if not dominant_threshold:
        dominant_threshold = config["train_val_split"]["dominant_threshold"]
    assert isinstance(dominant_threshold, float), "Invalid dominant threshold."

    run_manifest_generator(
        dst_dataset_path,
        dst_dataset_path,
        dst_properties_path,
        train_size,
        randomness,
        target_classes,
        dominant_threshold,
    )


def calculate_weight_cross_entropy(species_composition_path, species_labels_path):
    with open(species_composition_path, "r") as species_f:
        species_data = json.load(species_f)

    with open(species_labels_path, "r") as labels_f:
        labels_data = json.load(labels_f)

    species_names  = list(labels_data.values())

    species_counts = []

    for species in species_names:
        species_counts.append(species_data[species])
    counts_tensor = torch.tensor(species_counts, dtype=torch.float)

    inv_freq = 1.0 / counts_tensor
    weights = inv_freq / inv_freq.sum() * len(inv_freq)
    return weights