import os
import torch
from pipeline.utility import mobile_net_v3_large_builder, get_device
from thop import profile

device = get_device()

base_folder = "./models/"

# for model_file in sorted(os.listdir(base_folder)):
#     if not model_file.endswith(".pth"):
#         continue
#     model_name = model_file.replace(".pth", "")
model = mobile_net_v3_large_builder(device, path="/home/tom-maverick/Desktop/mobilenet_v3_large_100_hg_bird_insect_continue.pth")
input = torch.randn(1, 3, 224, 224).to(device)
macs, param = profile(model, inputs=(input, ), verbose=False)
# print(f"{model_name}: macs: {macs} | param: {param}")
print(f"macs: {macs} | param: {param}")