import os
import shutil
from tqdm import tqdm
from typing import Optional
from scripts.analysis.identifying_dominant_species import identifying_dominant_species
from dataset_builder.utility import FailedOperation


def species_path_extract(data_path: str) -> list[str]:
    """
    Extracts paths to species directories within a given dataset directory.
    
    Args:
        data_path: The base directory containing class folders.

    Returns:
        List[str]: A list of full paths to species directory.

    Raises:
        FileNotFoundError: If the given directory does not exists or is not a directory.
    """
    species_folder_path: list[str] = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory not found: {data_path}")

    for class_entry in os.scandir(data_path):
        if class_entry.is_dir():
            class_path = class_entry.path
            for species_entry in os.scandir(class_path):
                if species_entry.is_dir():
                    species_folder_path.append(species_entry.path)
    return species_folder_path


def copy_file(
    species_paths: list[str],
    out_dir: str,
    other_dir: str,
    isOther: bool,
    message: Optional[str] = None,
):
    """
    Copies files from species directories into a structured output directory.

    Args:
        species_paths: List of species directory paths.
        out_dir: The output directory where files should be copied.
        other_dir: The output directory where files belong to "Other".
        is_other: Determines if species belong to the 'Other' category.
        other_dir: Name of the 'Other' category folder. Defaults to "Other".
        message: Message to display before processing. Defaults to None.
    """
    if message:
        print(message)
    
    for species in tqdm(species_paths,f"Copying species to {other_dir if isOther else out_dir}"):
        species_parts = species.split(os.sep)
        if "species_lists" in species_parts:
            continue

        if isOther:
            species_base = species_parts[-1]
            species_out_dir = os.path.join(other_dir, species_base)
        else:
            species_base = os.path.join(species_parts[-2], species_parts[-1])
            species_out_dir = os.path.join(out_dir, species_base)

        os.makedirs(species_out_dir, exist_ok=True)

        for file in os.listdir(species):
            src = os.path.join(species, file)
            dst = os.path.join(species_out_dir, file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)


def run_other_dataset_builder_big(config):
    src_dir = config["paths"]["src_dataset"]
    inter_dir = config["paths"]["inter_dataset"]
    output_dir = config["paths"]["dst_dataset"]
    other_subdir = os.path.join(output_dir, "Other")

    os.makedirs(other_subdir, exist_ok=True)

    inter_dir_paths = species_path_extract(inter_dir)
    inter_species = [os.path.basename(f) for f in inter_dir_paths]

    src_dir_paths = species_path_extract(src_dir)
    other_folder_path = [f for f in src_dir_paths if os.path.basename(f) not in inter_species]

    message = f"Stage 1: Copying species not in {os.path.basename(inter_dir)} to 'Other'"
    copy_file(other_folder_path, output_dir, other_subdir, True, message)

    message = f"Stage 2: Copying species in {os.path.basename(inter_dir)} to their new location"
    copy_file(inter_dir_paths, output_dir, other_subdir, False, message)


def run_other_dataset_builder_small(config):
    src_dir = config["paths"]["inter_dataset"]
    dst_dir = config["paths"]["dst_dataset_small"]
    properties_path = os.path.join(config["paths"]["output_dir"], f"{os.path.basename(src_dir)}_properties.json")
    other_subdir = os.path.join(dst_dir, "Other")

    os.makedirs(other_subdir, exist_ok=True)

    threshold = config.get("train_val_split", {}).get("dominant_threshold", 0.5)
    included_classes = config["train_val_split"]["included_classes"]

    # Dynamically compute dominant species
    dominant_species_data = identifying_dominant_species(properties_path, threshold, included_classes)
    print(f"Dominant species: {dominant_species_data}")
    if dominant_species_data is None:
        raise FailedOperation("Could not compute dominant species")

    dominant_set = {
        os.path.join(class_name, species)
        for class_name, species_list in dominant_species_data.items()
        for species in species_list
    }

    src_species_paths = species_path_extract(src_dir)

    dominant_paths = []
    other_paths = []

    for path in src_species_paths:
        rel_path = os.path.relpath(path, src_dir)
        if rel_path in dominant_set:
            dominant_paths.append(path)
        else:
            other_paths.append(path)

    print(f"Dominant species: {len(dominant_paths)}")
    print(f"Other species: {len(other_paths)}")

    message = "Stage 1: Copying dominant species to class folders"
    copy_file(dominant_paths, dst_dir, other_subdir, isOther=False, message=message)

    message = "Stage 2: Copying non-dominant species to 'Other'"
    copy_file(other_paths, dst_dir, other_subdir, isOther=True, message=message)