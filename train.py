import json
import os
import time

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from torchvision import models, transforms  # type: ignore
from utility import CustomDataset, model_builder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

with open("./data/haute_garonne_other/dataset_species_labels.json") as file:
    species_labels = json.load(file)

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 15
NUM_SPECIES = len(species_labels.keys())

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

train_dataset = CustomDataset(
    txt_file="./data/haute_garonne_other/train.txt",
    root_dir=".",
    transform=transform_train,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

model = model_builder(models, NUM_SPECIES, device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train phase
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0

    start = time.perf_counter()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    train_epoch_loss = train_loss / len(train_loader.dataset)
    train_epoch_acc = train_correct.double() / len(train_loader.dataset)

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

    val_epoch_loss = val_loss / len(train_loader.dataset)
    val_epoch_acc = val_correct.double() / len(train_loader.dataset)

    scheduler.step()
    end = time.perf_counter()

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
        f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}, "
        f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}"
    )
    print(f"Epoch time: {end - start}s")

    torch.save(model.state_dict(), "mobilenet_v3_large.pth")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,  # Model being run
        dummy_input,  # Model input
        "mobilenet_v3_large.onnx",  # Output ONNX filename
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