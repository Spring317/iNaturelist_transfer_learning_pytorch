import json

import torch  # type: ignore
from dataset_builder.core import load_config
from tabulate import tabulate
from termcolor import colored

from pipeline.comparing import diff_models, get_model_summary, plot_layer_sparsity
from pipeline.utility import get_device

device = get_device(True)

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

config = load_config("./config.yaml")

BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 30
NUM_SPECIES = len(species_labels.keys())
DOM_THRESHOLD = config["train_val_split"]["dominant_threshold"]


original_model = torch.load("/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.pth", map_location=device, weights_only=False)
original_model.eval()
pruned_model = torch.load("/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds/mobilenet_v3_large_50_prune_30.pth", map_location=device, weights_only=False)
pruned_model.eval()

original_summary = get_model_summary(original_model)
pruned_summary = get_model_summary(pruned_model)
diff = diff_models(original_summary, pruned_summary)

colored_diff = []
for d in diff:
    orig_dims = d["orig_shape"]
    pruned_dims = d["pruned_shape"]
    dim_diff = []
    for o, p in zip(orig_dims, pruned_dims):
        if o != p:
            dim_diff.append(colored(str(p), "yellow"))
        else:
            dim_diff.append(str(p))

    if len(pruned_dims) > len(orig_dims):
        for i in range(len(orig_dims), len(pruned_dims)):
            dim_diff.append(colored(str(pruned_dims[i]), "yellow"))

    pruned_shaped_highlighted = "(" + ", ".join(dim_diff) + ")"

    row = {
        "layer": d["layer"],
        "orig_shape": str(orig_dims),
        "pruned_shape": pruned_shaped_highlighted
    }
    colored_diff.append(row)

print(tabulate(colored_diff, headers="keys"))
# plot_layer_sparsity(diff)