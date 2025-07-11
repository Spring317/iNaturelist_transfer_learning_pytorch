import json
import pandas as pd
import torch  # type: ignore
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from pipeline.utility import (
    get_device,
    get_support_list,
    generate_report,
    mobile_net_v3_large_builder,
    convnext_large_builder,
    manifest_generator_wrapper,
)
from sklearn.metrics import accuracy_score, f1_score, recall_score
from pipeline.dataset_loader import CustomDataset

def load_model_from_checkpoint(model_path, device):
    """Load model from checkpoint, handling different save formats."""
    try:
        # First try to load as a full model
        model = torch.load(model_path, weights_only=False, map_location=device)
        print("Loaded full model successfully")
        return model
    except Exception as e:
        print(f"Failed to load as full model: {e}")
        
        try:
            # Try to load as checkpoint with state_dict
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Checkpoint contains state_dict and metadata
                num_classes = checkpoint["model_state_dict"]["classifier.2.weight"].shape[0]
                model = convnext_large_builder(device, num_outputs=num_classes, start_with_weight=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded model from checkpoint with {num_classes} classes")
                return model
                
            elif isinstance(checkpoint, dict) and "classifier.2.weight" in checkpoint:
                # Checkpoint is a state_dict directly
                num_classes = checkpoint["classifier.2.weight"].shape[0]
                model = convnext_large_builder(device, num_outputs=num_classes, start_with_weight=False)
                model.load_state_dict(checkpoint)
                print(f"Loaded model from state_dict with {num_classes} classes")
                return model
            else:
                raise RuntimeError("Unrecognized checkpoint format")
                
        except Exception as e2:
            raise RuntimeError(f"Failed to load model from {model_path}. Tried both full model and checkpoint formats. Errors: {e}, {e2}")

# NNUM_SPECIES = 1000  # Adjust this based on your dataset
device = get_device()
BATCH_SIZE = 64
NUM_WORKERS = 12
NAME = "mobilenet_v3_large"

_, _, val_images, species_labels, species_composition = manifest_generator_wrapper(1.0)

# with open("./data/haute_garonne/dataset_species_labels_full_bird_insect.json") as file:
#     species_labels = json.load(file)

species_names = list(species_labels.values())
val_dataset = CustomDataset(val_images, train=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

total_support_list = get_support_list(
    species_composition, species_names
)

model_path = "models/convnext_full_nsect_best.pth"
device = get_device()
model = load_model_from_checkpoint(model_path, device)
model.eval()
val_loss, val_correct = 0.0, 0

all_preds: List[int] = []
all_labels: List[int] = []

print("Begin validating")
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validating", unit="Batch"):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

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
report_df = generate_report(
    all_labels, all_preds, species_names, total_support_list, float(accuracy)
)

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    # print(report_df)
    report_df.to_csv("./test.csv")
