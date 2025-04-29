import pandas as pd
from pipeline.comparing import plot_heat_map_feature_maps

# Re-import files after code execution environment reset
base_path = "./feature_maps/"
csv_files = [
    "mobilenet_v3_large_50_prune_0.csv",
    "mobilenet_v3_large_50_prune_5.csv",
    "mobilenet_v3_large_50_prune_10.csv",
    "mobilenet_v3_large_50_prune_15.csv",
    "mobilenet_v3_large_50_prune_20.csv",
    "mobilenet_v3_large_50_prune_25.csv",
    "mobilenet_v3_large_50_prune_30.csv",
    "mobilenet_v3_large_50_prune_35.csv",
    "mobilenet_v3_large_50_prune_40.csv",
    "mobilenet_v3_large_50_prune_45.csv",
    "mobilenet_v3_large_50_prune_50.csv",

    "mobilenet_v3_large_80_prune_0.csv",
    "mobilenet_v3_large_80_prune_5.csv",
    "mobilenet_v3_large_80_prune_10.csv",
    "mobilenet_v3_large_80_prune_15.csv",
    "mobilenet_v3_large_80_prune_20.csv",
    "mobilenet_v3_large_80_prune_25.csv",
    "mobilenet_v3_large_80_prune_30.csv",
    "mobilenet_v3_large_80_prune_35.csv",
    "mobilenet_v3_large_80_prune_40.csv",
    "mobilenet_v3_large_80_prune_45.csv",
    "mobilenet_v3_large_80_prune_50.csv",

    "mobilenet_v3_large_90_prune_0.csv",
    "mobilenet_v3_large_90_prune_5.csv",
    "mobilenet_v3_large_90_prune_10.csv",
    "mobilenet_v3_large_90_prune_15.csv",
    "mobilenet_v3_large_90_prune_20.csv",
    "mobilenet_v3_large_90_prune_25.csv",
    "mobilenet_v3_large_90_prune_30.csv",
    "mobilenet_v3_large_90_prune_35.csv",
    "mobilenet_v3_large_90_prune_40.csv",
    "mobilenet_v3_large_90_prune_45.csv",
    "mobilenet_v3_large_90_prune_50.csv",
]

dataframes = []
for file in csv_files:
    df = pd.read_csv(base_path + file)
    df["Prune"] = file.split("_")[-1].replace(".csv", "")
    dataframes.append(df)

plot_heat_map_feature_maps(dataframes[0:11], "mobilenet_v3_large_50.png", "Feature Map Size per Layer Prune Levels (MobileNetV3, dom=50)")
plot_heat_map_feature_maps(dataframes[11:22], "mobilenet_v3_large_80.png", "Feature Map Size per Layer Prune Levels (MobileNetV3, dom=80)")
plot_heat_map_feature_maps(dataframes[22:], "mobilenet_v3_large_90.png", "Feature Map Size per Layer Prune Levels (MobileNetV3, dom=90)")