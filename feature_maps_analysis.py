import pandas as pd
from pipeline.comparing import plot_heat_map_feature_maps

# Re-import files after code execution environment reset
base_path = "./"
csv_files = [
    "feature_map_sizes_mobilenet_v3_large_50_prune_0.csv",
    "feature_map_sizes_mobilenet_v3_large_50_prune_30.csv",
    "feature_map_sizes_mobilenet_v3_large_50_prune_35.csv",
    "feature_map_sizes_mobilenet_v3_large_50_prune_40.csv",
    "feature_map_sizes_mobilenet_v3_large_50_prune_45.csv",
    "feature_map_sizes_mobilenet_v3_large_50_prune_50.csv",
    # "feature_map_sizes_mobilenet_v3_large_80_prune_0.csv",
    # "feature_map_sizes_mobilenet_v3_large_80_prune_30.csv",
    # "feature_map_sizes_mobilenet_v3_large_80_prune_35.csv",
    # "feature_map_sizes_mobilenet_v3_large_80_prune_40.csv",
    # "feature_map_sizes_mobilenet_v3_large_80_prune_45.csv",
    # "feature_map_sizes_mobilenet_v3_large_80_prune_50.csv",
    # "feature_map_sizes_mobilenet_v3_large_90_prune_0.csv",
    # "feature_map_sizes_mobilenet_v3_large_90_prune_30.csv",
    # "feature_map_sizes_mobilenet_v3_large_90_prune_35.csv",
    # "feature_map_sizes_mobilenet_v3_large_90_prune_40.csv",
    # "feature_map_sizes_mobilenet_v3_large_90_prune_45.csv",
    # "feature_map_sizes_mobilenet_v3_large_90_prune_50.csv",
]

dataframes = []
for file in csv_files:
    df = pd.read_csv(base_path + file)
    df["Prune"] = file.split("_")[-1].replace(".csv", "")
    dataframes.append(df)

plot_heat_map_feature_maps(dataframes, "feature_maps_50.png", "Feature Map Size per Layer Across Different Prune Levels (MobileNetV3, dom=50)")