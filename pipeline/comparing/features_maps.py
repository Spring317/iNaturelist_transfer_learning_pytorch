import pandas as pd
import torch
from torch import nn
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns


def extracting_feature_maps(model: nn.Module) -> pd.DataFrame:
    """
    Analyzes the intermediate feature maps produced by a neural network model.

    This function registers forward hooks on all named layers of the given model 
    (excluding container layers like nn.Sequential), performs a forward pass using 
    a dummy input tensor of shape (1, 3, 224, 224), and collects shape and memory 
    usage statistics for each layer's output.

    The result is returned as a pandas DataFrame sorted by descending memory usage (KB).
    A summary row is also appended at the end showing the total memory used by all layers.

    Args:
        model (nn.Module): The PyTorch model to analyze.

    Returns:
        pd.DataFrame: A table containing:
            - Layer: The layer name.
            - Shape(s): List of shapes of output tensors from that layer.
            - Total Elements: Number of elements across all outputs.
            - Size (KB): Total memory footprint in kilobytes (float32 precision).
            - TOTAL row: A final summary row aggregating memory size over all layers.
    """
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                shapes = [output.shape]
            elif isinstance(output, (tuple, list)):
                shapes = [o.shape for o in output if isinstance(o, torch.Tensor)]
            else:
                shapes = []

            total_elements = sum(torch.tensor([], dtype=torch.float32).new_empty(s).numel() for s in shapes)
            bytes_per_element = 4  # float32
            total_kb = (total_elements * bytes_per_element) / 1024

            feature_data.append({
                "Layer": name,
                "Shape(s)": shapes,
                "Total Elements": total_elements,
                "Size (KB)": round(total_kb, 2)
            })
        return hook

    feature_data: List[Dict] = []
    model.eval()

    # register hooks
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and name:
            module.register_forward_hook(hook_fn(name))

    # pass in dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _ = model(dummy_input)

    # DataFrame
    df_features = pd.DataFrame(feature_data)
    df_features = df_features.sort_values(by="Size (KB)", ascending=False).reset_index(drop=True)
    total_size_kb = df_features["Size (KB)"].sum()

    # add summary row
    summary_row = pd.DataFrame([{
        "Layer": "TOTAL",
        "Shape": "",
        "Num Elements": "",
        "Size (KB)": round(total_size_kb, 2)
    }])

    # add the row to the DataFrame
    df_features = pd.concat([df_features, summary_row], ignore_index=True)
    return df_features


def plot_heat_map_feature_maps(dataframes: List[pd.DataFrame], fig_name: str, fig_title: str):
    """
    Plots a heatmap of memory usage (in KB) across model layers and pruning levels.

    This function takes in multiple DataFrames containing layer-wise feature map sizes 
    for different pruning levels, combines them, and creates a pivoted heatmap showing 
    memory usage per layer per pruning level. It also appends a summary text at the top 
    showing total memory per pruning level.

    Args:
        dataframes (List[pd.DataFrame]): 
            A list of DataFrames, each containing the columns:
                - "Layer": Name of the layer (string)
                - "Prune": Pruning level (float)
                - "Size (KB)": Memory usage of the layer’s output (float)
        fig_name (str): The filename (with path) to save the resulting heatmap figure.
        fig_title (str): The title to display at the top of the heatmap.

    Notes:
        - Rows where `Layer == "TOTAL"` (case-insensitive) are excluded from the heatmap,
          but their values are summarized and shown as text above the plot.
        - The heatmap uses a blue-green ("YlGnBu") color scale and annotates each cell
          with the actual memory value.
        - Output image is saved as a static PNG file via `plt.savefig(fig_name)`.
    """
    # combine and sort
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df["Prune"] = combined_df["Prune"].astype(int)
    combined_df.sort_values(by=["Layer", "Prune"], inplace=True)

    # save total row for later
    total_rows = combined_df[combined_df["Layer"].str.lower() == "total"]

    # filter out total row for heatmap
    filtered_df = combined_df[combined_df["Layer"].str.lower() != "total"]

    # pivot for heatmap
    pivot_df = filtered_df.pivot(index="Layer", columns="Prune", values="Size (KB)")

    # plot
    plt.figure(figsize=(16, 50))
    sns.heatmap(pivot_df, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={"label": "Size (KB)"})
    plt.title(fig_title, fontsize=16)
    plt.ylabel("Layer")
    plt.xlabel("Prune Level")
    plt.xticks(rotation=0)

    # add total row values as summary text
    totals = total_rows.sort_values("Prune")["Size (KB)"].values
    prune_levels = total_rows.sort_values("Prune")["Prune"].values
    summary_text = "Total (KB): " + ", ".join([f"{p}%: {s:.2f} KB" for p, s in zip(prune_levels, totals)])
    plt.figtext(0.5, 0.97, summary_text, ha="center", fontsize=10)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))  # leave space for the figtext
    plt.savefig(fig_name)