import json
import os
import pandas as pd
import torch  # type: ignore
import numpy as np
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_builder.core import load_config, validate_config
from dataset_builder.core.exceptions import ConfigError
from pipeline.utility import get_device, get_support_list, generate_report
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)
from pipeline.dataset_loader import CustomDataset
from pipeline.utility import manifest_generator_wrapper
try:
    config = load_config("./config.yaml")
    validate_config(config)
except ConfigError as e:
    print(e)
    exit()

dominant_threshold = [0.5, 0.8, 0.9]

device = get_device()
BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 1
NAME = "mobilenet_v3_large"
POST_FIX = "_pruned_train"
# NUM_SPECIES = len(species_labels.keys())

for threshold in dominant_threshold:
    manifest_generator_wrapper(threshold)

    with open("./data/haute_garonne/dataset_species_labels.json") as file:
        species_labels = json.load(file)

    species_names = list(species_labels.values())
    val_dataset = CustomDataset("./data/haute_garonne", train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    total_support_list = get_support_list("./data/haute_garonne/species_composition.json", species_names)

    for base_model in sorted(os.listdir("./models")):
        if not base_model.endswith(".pth"):
            continue
        model_name = base_model.replace(".pth", "")
        model_name_list = model_name.split("_")
        dom_threshold = int(model_name_list[3])
        if not dom_threshold == int(threshold * 100):
            continue
        print(f"Validating: {base_model}")
        dom_threshold = int(model_name_list[3]) / 100  # type: ignore
        prune_threshold = int(model_name_list[-1])

        model = torch.load(os.path.join("./models", base_model), map_location=device, weights_only=False)
        model = model.to(device)


        model.eval()
        val_loss, val_correct = 0.0, 0

        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        print("Begin validating")
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", unit="Batch"):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                # print("Prediction counts:", np.unique(all_preds, return_counts=True))
                all_labels.extend(labels.cpu().numpy())
                print("Label counts:", np.unique(all_labels, return_counts=True))

        accuracy = accuracy_score(all_labels, all_preds)
        weighted_recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Weighted Recall: {weighted_recall:.4f}")
        print(f"Weighted Average F1-Score: {f1:.4f}")
        report_df = generate_report(all_labels, all_preds, species_names, total_support_list, float(accuracy))

        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            # print(report_df)
            report_df.to_csv(f"./threshold_reports/mobilenet_v3_large_{threshold* 100:.0f}_prune_{prune_threshold}.csv")