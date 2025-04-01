import json
import os
import time

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from tqdm import tqdm
from torch.utils.data import DataLoader  # type: ignore
from torchvision import models, transforms  # type: ignore
from dataset_builder.core import load_config
from utility import CustomDataset, model_builder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

config = load_config("./config.yaml")

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 30
NUM_SPECIES = len(species_labels.keys())
DOM_THRESHOLD = config["train_val_split"]["dominant_threshold"]

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

print("Loading training, validation datasets")
train_dataset = CustomDataset(
    data_path="./data/haute_garonne/train.parquet",
    root_dir=".",
    transform=transform_train,
)

val_dataset = CustomDataset(
    data_path="./data/haute_garonne/val.parquet", root_dir=".", transform=transform_val
)

print("Creating DataLoaders")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

model = model_builder(models, NUM_SPECIES, device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print("Begin training")
for epoch in range(NUM_EPOCHS):
    # Train phase
    model.train()
    train_loss, train_correct = 0.0, 0

    start = time.perf_counter()
    loop = tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Training")
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

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
        loop.set_postfix(loss=f"{loss.item():.3f}")

    train_epoch_loss = train_loss / float(len(train_loader.dataset))
    train_epoch_acc = train_correct.double() / float(len(train_loader.dataset))

    # Validation phase
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_epoch_loss = val_loss / float(len(train_loader.dataset))
    val_epoch_acc = val_correct.double() / float(len(train_loader.dataset))

    scheduler.step()
    end = time.perf_counter()

    print(
        f"[Epoch {epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}, "
        f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}"
    )
    print(f"Epoch time: {end - start}s")

    print("Saving model")
    torch.save(model, f"./models/mobilenet_v3_large_{DOM_THRESHOLD * 100:.0f}.pth")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    model_to_export = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.onnx.export(
        model_to_export,  # Model being run
        dummy_input,  # Model input
        f"./models/mobilenet_v3_large.onnx_{DOM_THRESHOLD * 100:.0f}",  # Output ONNX filename
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