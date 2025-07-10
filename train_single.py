import json
import time
import os
import re

import torch  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from pipeline.dataset_loader import CustomDataset
from pipeline.training import save_model, train_one_epoch, train_validate
from pipeline.utility import calculate_weight_cross_entropy, manifest_generator_wrapper, get_device, mobile_net_v3_large_builder, convnext_large_builder

def find_latest_checkpoint(model_name, models_dir="models"):
    """Find the latest checkpoint file for the given model name."""
    if not os.path.exists(models_dir):
        return None, 0
    
    checkpoint_files = []
    pattern = f"{model_name}_epoch_(\d+).pth"
    
    for file in os.listdir(models_dir):
        match = re.match(pattern, file)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((epoch_num, os.path.join(models_dir, file)))
    
    if not checkpoint_files:
        return None, 0
    
    # Sort by epoch number and return the latest
    checkpoint_files.sort(key=lambda x: x[0])
    latest_epoch, latest_file = checkpoint_files[-1]
    return latest_file, latest_epoch

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and return the epoch number."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If checkpoint only contains model weights
        model.load_state_dict(checkpoint)
        return 0  # Can't determine epoch from model-only checkpoint
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get epoch number
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}")
    return epoch

_, train, val, _, _=  manifest_generator_wrapper(0.5, export=True)  # type: ignore
print("=========================================")
device = get_device()
print("=========================================")

with open("haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

BATCH_SIZE = 64
NUM_WORKERS = 8
NUM_EPOCHS = 50
NUM_SPECIES = len(species_labels.keys())
NAME = "convnext_full_nsect"
ENABLE_EXPERIMENTAL_HYPERPARAM_TUNING = True
INPUT_SIZE = 160  # Changed from 224 to 160

model = convnext_large_builder(device, num_outputs=NUM_SPECIES, start_with_weight=True, input_size=INPUT_SIZE)
train_dataset = CustomDataset(train, train=True)
val_dataset = CustomDataset(val, train=False)
torch.save(train_dataset, "train_dataset.pth")
torch.save(val_dataset, "val_dataset.pth")

if ENABLE_EXPERIMENTAL_HYPERPARAM_TUNING:
    warmup_epochs = 5

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - warmup_epochs)
        ],
        milestones=[warmup_epochs]
    )
    weights = calculate_weight_cross_entropy("./haute_garonne/species_composition.json", "./haute_garonne/dataset_species_labels.json")
    weights = weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
else:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)  #type: ignore
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  #type: ignore

# Check for existing checkpoint
checkpoint_path, last_epoch = find_latest_checkpoint(NAME)
start_epoch = 0
if checkpoint_path:
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

best_acc = -1.0
best_f1 = -1.0
for epoch in range(start_epoch, NUM_EPOCHS):
    start = time.perf_counter()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, macro_f1 = train_validate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val acc: {val_acc:.4f} Val F1: {macro_f1:.4f}")
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_f1 = macro_f1
        save_model(model, f"{NAME}_best", "models", device, (INPUT_SIZE, INPUT_SIZE))
        print(f"New best model saved!")

    # Save periodic checkpoint every N epochs
    # if (epoch + 1) % 10 == 0:  # Save every 10 epochs
    # Create checkpoint with full training state
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'best_f1': best_f1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'macro_f1': macro_f1
    }
    torch.save(checkpoint, f"models/{NAME}_epoch_{epoch+1}.pth")

    end = time.perf_counter()
    print(f"Total time: {end - start:.2f}s")
print(f"Best accuracy: {best_acc} with F1-score: {best_f1}")