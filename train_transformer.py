import time

import torch  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from pipeline.dataset_loader import CustomDataset
from pipeline.training import save_model, train_one_epoch, train_validate
from pipeline.utility import calculate_weight_cross_entropy, manifest_generator_wrapper, get_device, mobile_net_v3_large_builder, convnext_large_builder

_, train, val, species_labels, species_composition =  manifest_generator_wrapper(1.0)  # type: ignore
print()
device = get_device()
print()


BATCH_SIZE = 2
NUM_WORKERS = 8
NUM_EPOCHS = 50
NUM_SPECIES = len(species_labels.keys())
NAME = "hiera_full_inat_bird_insect"

model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
old_linear_layer = model.head.projection
model.head.projection = torch.nn.Linear(old_linear_layer.in_features, NUM_SPECIES, bias=True)
model.to(device)

train_dataset = CustomDataset(train, train=True)
val_dataset = CustomDataset(val, train=False)

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
weights = calculate_weight_cross_entropy(species_composition, species_labels)
weights = weights.to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

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
for epoch in range(NUM_EPOCHS):
    start = time.perf_counter()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, macro_f1 = train_validate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val acc: {val_acc:.4f} Val F1: {macro_f1:.4f}")
    if val_acc > best_acc:
        start_save = time.perf_counter()
        best_acc = val_acc
        best_f1 = macro_f1
        save_model(model, f"{NAME}", "models", device, (224, 224))
        end_save = time.perf_counter()
        print(f"Save time: {end_save - start_save:.2f}s")
    end = time.perf_counter()
    print(f"Total time: {end - start:.2f}s")
print(f"Best accuracy: {best_acc} with F1-score: {best_f1}")