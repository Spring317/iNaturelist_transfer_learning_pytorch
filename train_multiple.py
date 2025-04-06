import json
import os
import time

import gc
import torch  # type: ignore
from utility import save_model, get_device, build_dataloaders, train_one_epoch, validate
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


device = get_device()


BATCH_SIZE = 96
NUM_WORKERS = 16
NUM_EPOCHS = 15
# NUM_SPECIES = len(species_labels.keys())
NAME = "mobilenet_v3_large"
MODEL_PATH = "./models/"
DOM_THRESHOLDS = [50, 80, 90]
print(f"Using {NUM_WORKERS} workers")

for threshold in DOM_THRESHOLDS:
    print("\n\n")
    run_manifest_generator(
        dst_dataset_path,
        dst_dataset_path,
        dst_properties_path,
        train_size,
        randomness,
        target_classes,
        threshold / 100,
    )
    print("\n\n")

    train_loader, val_loader = build_dataloaders(
        "./data/haute_garonne", BATCH_SIZE, NUM_WORKERS
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
        model = torch.load(
            os.path.join(MODEL_PATH, base_model),
            map_location=device,
            weights_only=False,
        )
        model = model.to(device)

        with open("./data/haute_garonne/dataset_species_labels.json") as file:
            species_labels = json.load(file)

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
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()
            end = time.perf_counter()
            print(
                f"[Epoch {epoch + 1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {end - start:.2f}s"
            )
            if val_acc > best_acc:
                best_acc = val_acc
                print("Saving model...")
                start_save = time.perf_counter()
                save_model(
                    model,
                    f"{NAME}_{dom_threshold * 100:.0f}_prune_{prune_threshold}",
                    device,
                )
                end_save = time.perf_counter()
                print(f"Save time: {end_save - start_save:.2f}s", end="\n\n")
    del train_loader
    del val_loader
    torch.cuda.empty_cache()

    gc.collect()
