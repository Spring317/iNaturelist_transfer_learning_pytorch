import json

import torch  # type: ignore
import pandas as pd
import numpy as np
import torch.nn as nn  # type: ignore
import torch.nn.utils.prune as prune
from collections import Counter
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from typing import List
from torch.utils.data import DataLoader, Dataset  # type: ignore
from torchvision import models, transforms  # type: ignore
from utility import CustomDataset, model_builder
from dataset_builder.core import load_config

def apply_global_prunning(model, amount=0.2):
    linear_counter = 0
    conv2d_counter = 0
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d ):
            linear_counter+=1
            parameters_to_prune.append((module, "weight"))
        if isinstance(module, nn.Linear):
            conv2d_counter+=1
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    print(f"Global pruning applied: {amount*100:.1f}% of weights set to zero.")

    return linear_counter, conv2d_counter

def remove_pruning_reparam(model):

    for module in model.modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, "weight")
            except ValueError:
                print("Failed to remove Linear")
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except ValueError:
                print("Failed to remove Conv2d")

def apply_structured_pruning(model, amount=0.3):
    print(f"Applying structured pruning (L2 norm, {amount*100:.0f}% of output channels)...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Skip depthwise convs (groups=in_channels)
            if module.groups != 1:
                continue
            try:
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                prune.remove(module, 'weight')
                print(f"Pruned: {name}")
            except Exception as e:
                print(f"Skipped {name}: {e}")


def print_layer_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_zero = torch.sum(module.weight == 0).item()
            total = module.weight.nelement()
            print(f"{name}: {100. * num_zero / total:.2f}% sparsity")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

config = load_config("./config.yaml")

species_names = list(species_labels.values())

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_SPECIES = len(species_labels.keys())
DOM_THRESHOLD = config["train_val_split"]["dominant_threshold"]

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

print("Loading training, validation datasets")
train_dataset = CustomDataset(
    data_path="./data/haute_garonne/train.parquet", root_dir=".", transform=transform_val  # The transform is not important
)

val_dataset = CustomDataset(
    data_path="./data/haute_garonne/val.parquet", root_dir=".", transform=transform_val
)

print("Creating DataLoaders")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

all_train_labels = []
for _, labels in train_loader:
    all_train_labels.extend(labels.cpu().numpy())
train_counts = Counter(all_train_labels)
train_support_list = [train_counts.get(i, 0) for i in range(len(species_names))]

# model = model_builder(models, NUM_SPECIES, device, is_eval=True)
# model = model.to(device)
model = torch.load(f"./models/mobilenet_v3_large_{DOM_THRESHOLD * 100:.0f}.pth", map_location=device)

apply_structured_pruning(model)
# linear, conv2d = apply_global_prunning(model, amount=0.2)
# print(f"Linear {linear}, Conv2D {conv2d}")
# remove_pruning_reparam(model)

# print_layer_sparsity(model)

print("Saving model")
torch.save(model, f"./models/mobilenet_v3_large_{DOM_THRESHOLD * 100:.0f}_pruned.pth")
model.eval()
dummy_input = torch.randn(1, 3, 224, 224, device=device)
model_to_export = model.module if isinstance(model, torch.nn.DataParallel) else model
torch.onnx.export(
    model_to_export,  # Model being run
    dummy_input,  # Model input
    f"./models/mobilenet_v3_large.onnx_{DOM_THRESHOLD * 100:.0f}_pruned",  # Output ONNX filename
    export_params=True,  # Store trained parameter weights
    opset_version=14,  # ONNX opset version
    do_constant_folding=True,  # Perform constant folding optimization
    input_names=["input"],  # Model input name
    output_names=["output"],  # Model output name
    dynamic_axes={
        "input": {0: "batch_size"},  # Variable batch size
        "output": {0: "batch_size"},
    },
)

model.eval()

val_loss, val_correct = 0.0, 0

all_preds: List[np.ndarray] = []
all_labels: List[np.ndarray] = []

print("Begin validating")
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
weighted_recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")

print(f"Validation accuracy: {accuracy:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted Average F1-Score: {f1:.4f}")
report_dict = classification_report(all_labels, all_preds,target_names=species_names, digits=4, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc[species_names, "train_support"] = train_support_list
report_df.loc[species_names, "train_support"] = report_df.loc[species_names, "train_support"].astype(int)
species_df = report_df.loc[species_names].copy()
summary_df = report_df.drop(index=species_names).copy()
species_df = species_df.sort_values(by="f1-score", ascending=True)
report_df = pd.concat([species_df, summary_df])
# report_df.drop(columns=["support"], inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    report_df.to_csv(f"./reports/mobilenet_v3_large_{DOM_THRESHOLD * 100:.0f}_prunned.csv")