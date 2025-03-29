import os
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset  # type: ignore
from PIL import Image  # type: ignore
from typing import List, Tuple

class CustomDataset(Dataset):
    def __init__(self, data_path, root_dir, transform=None):
        self.image_labels = load_manifest_parquet(data_path)
        self.root_dir = root_dir
        self.transform = transform

        # with open(txt_file, "r") as file:
        #     lines = file.readlines()
        #     for line in lines:
        #         path, label = line.strip().split(":")
        #         self.image_labels.append((path, int(label)))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        img_path, label = self.image_labels[index]
        img_full_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def model_builder(models, num_species, device, is_eval=False):
    if is_eval:
        model = models.mobilenet_v3_large(weights=None)
    else:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_species)
    model = model.to(device)
    return model


def load_manifest_parquet(path: str) -> List[Tuple[str, int]]:
    """
    Loads a Parquet dataset manifest and returns it as a list of tuples.
    """
    df = pd.read_parquet(path)
    return list(df.itertuples(index=False, name=None))