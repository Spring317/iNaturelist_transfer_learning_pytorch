import json

import pandas as pd
from dataset_builder.core import load_config

from torch.utils.data import DataLoader  # type: ignore

from pipeline.dataset_loader import CustomDataset
from pipeline.training import validation_onnx

with open("./data/haute_garonne/dataset_species_labels.json") as file:
    species_labels = json.load(file)

config = load_config("./config.yaml")

species_names = list(species_labels.values())

BATCH_SIZE = 64
NUM_WORKERS = 12

val_dataset = CustomDataset(
    data_path="./data/haute_garonne/val.parquet", train=False, img_size=(299, 299)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

onnx_model_path = "/home/tom-maverick/Documents/Final Results/InceptionV3_HG_onnx/inceptionv3_50.onnx"
report_df = validation_onnx(onnx_model_path, val_loader, species_names, device="cuda")

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    # print(report_df)
    report_df.to_csv("./reports/test.csv")