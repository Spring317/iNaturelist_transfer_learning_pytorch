from torchsummary import summary
import torchvision.models as models
import torch

model = torch.load(
    "/home/quydx/iNaturelist_transfer_learning_pytorch/models/convnext_full_insect_best.pth"
)
summary(model, input_size=(3, 160, 160))
