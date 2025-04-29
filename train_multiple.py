import os
import time

import gc
import torch  # type: ignore
from torch.utils.data import DataLoader
from pipeline.utility import mobile_net_v3_large_builder, get_device
from pipeline.dataset_loader import CustomDataset
from pipeline.utility import manifest_generator_wrapper
from pipeline.training import train_one_epoch, train_validate, save_model


device = get_device()


BATCH_SIZE = 64
NUM_WORKERS = 16
NUM_EPOCHS = 15
NAME = "mobilenet_v3_large"
MODEL_PATH = "./models/global_prune/"
DOM_THRESHOLDS = [80, 90]
print(f"Using {NUM_WORKERS} workers")

for threshold in DOM_THRESHOLDS:
    print("\n\n")
    _, train_images, val_images, _, _ = manifest_generator_wrapper(threshold / 100) # type: ignore
    print("\n\n")

    train_dataset = CustomDataset(train_images, train=True)
    val_dataset = CustomDataset(val_images, train=False)
    
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

    for base_model in sorted(os.listdir(MODEL_PATH)):
        if not base_model.endswith(".pth"):
            continue
        model_name = base_model.replace(".pth", "")
        model_name_list = model_name.split("_")
        dom_threshold = int(model_name_list[3])
        if not dom_threshold == threshold:
            continue
        print(f"Training: {base_model}")
        dom_threshold = int(model_name_list[3]) / 100  # type: ignore
        prune_threshold = int(model_name_list[-1])
        model_path = os.path.join(MODEL_PATH, base_model)
        model = mobile_net_v3_large_builder(device, path=model_path)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_acc = -1.0
        for epoch in range(NUM_EPOCHS):
            start = time.perf_counter()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, macro_f1 = train_validate(model, val_loader, criterion, device)
            scheduler.step()
            print(f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val acc: {val_acc:.4f} Val F1: {macro_f1:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                print("Saving model...")
                start_save = time.perf_counter()
                save_model(
                    model,
                    f"{NAME}_{dom_threshold * 100:.0f}_prune_{prune_threshold}",
                    "model/new_model",
                    device,
                    (224, 224)
                )
                end_save = time.perf_counter()
                print(f"Save time: {end_save - start_save:.2f}s", end="\n\n")
            end = time.perf_counter()
            print("Total time: {end - start:.2f}s")

    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    gc.collect()
