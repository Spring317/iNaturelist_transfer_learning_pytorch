import torch  # type: ignore
import torch_pruning as tp  # type: ignore
import numpy as np
from pipeline.utility import get_device, mobile_net_v3_large_builder

device = get_device()

BASE_MODEL_PRUNED_PATH = "/home/tom-maverick/Documents/Final Results/MobileNetV3/"

example_inputs = torch.randn(1, 3, 224, 224).to(device)
model = mobile_net_v3_large_builder(device, path="../../Final Results/MobileNetV3/mobilenet_v3_large_90.pth")

pruning_ratio = {}

# Define the default and sensitive ratio
default_ratio = 0.7
sensitive_ratio = 0.1

for m in model.modules():
    if 'SqueezeExcitation' == m.__class__.__name__:
        print("sensitive")
        pruning_ratio[m] = sensitive_ratio

importance = tp.importance.MagnitudeImportance(p=1)  # L1 norm
ignored_layers = [
    model.classifier[0],
    model.classifier[3]
]
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs=example_inputs,
    importance=importance,
    pruning_ratio=0.5,  
    iterative_steps=1,  
    ignored_layers=ignored_layers,  #type: ignore
    global_pruning=True,
    isomorphic=False,
    pruning_ratio_dict=pruning_ratio
)
pruner.step()
torch.save(model,"./models/mobilenet_v3_large_exp_dict_prune_90.pth",)
