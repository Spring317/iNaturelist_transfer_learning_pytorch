import os

import torch  # type: ignore
import torch_pruning as tp  # type: ignore
import numpy as np
from pipeline.utility import get_device, mobile_net_v3_large_builder
from pipeline.pruning import one_shot_prune

device = get_device()

BASE_MODEL_PRUNED_PATH = "./models/base_model_trained"
prune_ratios = np.linspace(0.3, 0.5, num=5)

for base_model in sorted(os.listdir(BASE_MODEL_PRUNED_PATH)):
    print(f"Prunning: {base_model}")
    model_name = base_model.replace(".pth", "")
    model_name_list = model_name.split("_")
    dom_threshold = int(model_name_list[-1])

    model = mobile_net_v3_large_builder(device, path=os.path.join(BASE_MODEL_PRUNED_PATH, base_model))
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    print("Prune ratio: ", end="")
    for prune_ratio in prune_ratios:
        print(prune_ratio, end=" ")

        importance = tp.importance.MagnitudeImportance(p=1)  # L1 norm

        one_shot_prune(model, importance, example_inputs, prune_ratio, [model.classifier[3]])
        folder_name = f"./models/prune_threshold_{prune_ratio * 100:.0f}"
        os.makedirs(folder_name, exist_ok=True)
        torch.save(
            model,
            f"./{folder_name}/mobilenet_v3_large_{dom_threshold}_prune_{prune_ratio * 100:.0f}.pth",
        )
    print()
