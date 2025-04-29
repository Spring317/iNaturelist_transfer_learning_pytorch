import os

import torch  # type: ignore
import torch_pruning as tp  # type: ignore
from pipeline.utility import get_device, mobile_net_v3_large_builder

device = get_device()

BASE_MODEL_PRUNED_PATH = "/home/tom-maverick/Documents/Final Results/MobileNetV3/"
prune_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for base_model in sorted(os.listdir(BASE_MODEL_PRUNED_PATH)):
    if not base_model.endswith(".pth"):
        continue
    print(f"Prunning: {base_model}")
    model_name = base_model.replace(".pth", "")
    model_name_list = model_name.split("_")
    dom_threshold = int(model_name_list[-1])

    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    print("Prune ratio: ", end="")
    for prune_ratio in prune_ratios:
        model = mobile_net_v3_large_builder(device, path=os.path.join(BASE_MODEL_PRUNED_PATH, base_model))
        print(prune_ratio, end=" ")

        importance = tp.importance.MagnitudeImportance(p=1)  # L1 norm
        # ignored_layers = [
        #     m for m in model.modules()
        #     if isinstance(m, torch.nn.Conv2d) and m.groups != 1
        # ]
        # ignored_layers.append(model.classifier[3])

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=importance,
            pruning_ratio=prune_ratio,  
            iterative_steps=1,  
            ignored_layers=[model.classifier[3]],
            global_pruning=True,
            isomorphic=True,

        )
        pruner.step()
        folder_name = "./models/global_isomorphic_prune"
        os.makedirs(folder_name, exist_ok=True)
        torch.save(
            model,
            f"./{folder_name}/mobilenet_v3_large_{dom_threshold}_global_prune_{prune_ratio * 100:.0f}.pth",
        )
    print()
