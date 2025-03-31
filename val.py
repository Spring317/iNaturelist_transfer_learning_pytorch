import json

import torch  # type: ignore
import pandas as pd
import numpy as np
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

config = load_config("./config.yaml")

species_names = list(species_labels.values())

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 15
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

train_dataset = CustomDataset(
    data_path="./data/haute_garonne/train.parquet", root_dir=".", transform=transform_val  # The transform is not important
)

val_dataset = CustomDataset(
    data_path="./data/haute_garonne/val.parquet", root_dir=".", transform=transform_val
)

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

model = model_builder(models, NUM_SPECIES, device, is_eval=True)
model = model.to(device)
model.load_state_dict(torch.load(f"./models/mobilenet_v3_large_{DOM_THRESHOLD * 100:.0f}.pth", map_location=device))

model.eval()

val_loss, val_correct = 0.0, 0

all_preds: List[np.ndarray] = []
all_labels: List[np.ndarray] = []

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
# sorted_report_df = report_df.sort_values(by="f1-score")
report_df.drop(columns=["support"], inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(report_df)