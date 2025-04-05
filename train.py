import json
import os
import time

import torch  # type: ignore
from utility import save_model, model_builder, get_device, build_dataloaders, train_one_epoch, validate
from dataset_builder import run_manifest_generator
from dataset_builder.core import load_config, validate_config
from dataset_builder.core.exceptions import ConfigError


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
dst_properties_path = os.path.join(output_path, f"{dst_dataset_name}_composition.json")
train_size = config["train_val_split"]["train_size"]
randomness = config["train_val_split"]["random_state"]
dominant_threshold = config["train_val_split"]["dominant_threshold"]

run_manifest_generator(
    dst_dataset_path,
    dst_dataset_path,
    dst_properties_path,
    train_size,
    randomness,
    target_classes,
    dominant_threshold,
)

device = get_device()

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 1
NUM_SPECIES = len(species_labels.keys())
NAME = "mobilenet_v3_large"

model = model_builder(NUM_SPECIES)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

train_loader, val_loader = build_dataloaders("./data/haute_garonne", BATCH_SIZE, NUM_WORKERS)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_acc = -1.0
for epoch in range(NUM_EPOCHS):
    start = time.perf_counter()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    end = time.perf_counter()
    print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {end - start:.2f}s |", end="")
    start = time.perf_counter()
    if val_acc > best_acc:
        best_acc = val_acc
        save_model(model, f"{NAME}_{dominant_threshold * 100:.0f}", device)
    end = time.perf_counter()
    print(f"Save time: {end - start:.2f}s")
