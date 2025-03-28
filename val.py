import json

import torch  # type: ignore
import pandas as pd
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset  # type: ignore
from torchvision import models, transforms  # type: ignore
from utility import CustomDataset, model_builder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

with open("./data/haute_garonne_other/dataset_species_labels.json") as file:
    species_labels = json.load(file)

species_names = list(species_labels.values())

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 15
NUM_SPECIES = len(species_labels.keys())

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

val_dataset = CustomDataset(
    txt_file="./data/haute_garonne_other/val.txt", root_dir=".", transform=transform_val
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

model = model_builder(models, NUM_SPECIES, device, is_eval=True)
model.load_state_dict(torch.load("./mobilenet_v3_large_80.pth", map_location=device))

model.eval()

val_loss, val_correct = 0.0, 0

all_preds, all_labels = [], []

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
sorted_report_df = report_df.sort_values(by="f1-score")
print(sorted_report_df)