import json
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch  # type: ignore
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline.dataset_loader import CustomDataset
from pipeline.utility import (
    generate_report,
    get_device,
    get_support_list,
    mobile_net_v3_large_builder,
    manifest_generator_wrapper
)

device = get_device()
BATCH_SIZE = 256
NUM_WORKERS = 16
NAME = "mobilenet_v3_large_90_post_hoc_approach_baseline_test_second_run"

all_images, train_images, val_images, species_labels, _ = manifest_generator_wrapper(0.9)

with open("./data/haute_garonne/dataset_species_labels_full.json") as file:
    species_labels_full: Dict[str, int] = json.load(file)

species_names_dominant = list(species_labels.values())
species_labels_dominant = []
species_labels_non_dominant = []
for label, name in species_labels_full.items():
    if name in species_names_dominant:
        species_labels_dominant.append(int(label))
    else:
        species_labels_non_dominant.append(int(label))


val_dataset = CustomDataset(train_images, train=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

total_support_list = get_support_list("./data/haute_garonne/species_composition.json", species_names_dominant)

model = mobile_net_v3_large_builder(device, path="/home/tom-maverick/Desktop/baseline/mobilenet_v3_large_100_baseline.pth")

model.eval()
val_loss, val_correct = 0.0, 0

all_preds: List[int] = []
all_labels: List[int] = []

print("Begin validating")
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validating", unit="Batch"):
        images = images.to(device)
        logits = model(images)
        distinct_logits = logits[:, species_labels_dominant]
        grouped_logits = logits[:, species_labels_non_dominant]
        other_logits = torch.logsumexp(grouped_logits, dim=1, keepdim=True)
        final_logits = torch.cat([distinct_logits, other_logits], dim=1)
        
        probs = torch.softmax(final_logits, dim=1)
        preds = torch.argmax(probs, dim=1)


        all_preds.extend(preds.cpu().numpy())
        # print("Prediction counts:", np.unique(all_preds, return_counts=True))
        all_labels.extend(labels.cpu().numpy())
        # print("Label counts:", np.unique(all_labels, return_counts=True))

accuracy = accuracy_score(all_labels, all_preds)
weighted_recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")

print(f"Validation accuracy: {accuracy:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted Average F1-Score: {f1:.4f}")
report_df = generate_report(all_labels, all_preds, species_names_dominant, total_support_list, float(accuracy))

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    # print(report_df)
    report_df.to_csv(f"./{NAME}.csv")

# cm = confusion_matrix(all_labels, all_preds, labels=list(map(int, species_labels.keys())))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(species_labels.values()))
# fig, ax = plt.subplots(figsize=(40, 40))
# disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=True)
# plt.title("Confusion Matrix (Monte Carlo Simulation)")
# plt.tight_layout()
# plt.savefig(f"MonteCarloConfusionMatrix_{NAME}.png")