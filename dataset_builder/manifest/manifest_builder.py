import json
import os
from typing import Dict, List, Optional, Set, Tuple

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset_builder.core.exceptions import FailedOperation
from dataset_builder.core.utility import save_manifest_parquet, write_data_to_json
from dataset_builder.manifest.identifying_dominant_species import (
    _identifying_dominant_species,
)


def _save_data_manifest(file_path: str, data: List[Tuple[str, int]]):
    """
    Saves a list of image paths and species IDs to a file.

    Each image path is written alongside its corresponding species ID.

    Args:
        file_path: The file path where the data will be saved.
        data: A list of tuples where each tuple contains an image path and its corresponding species ID.
    """
    with open(file_path, "w") as file:
        for img_path, species_id in data:
            file.write(f"{img_path}: {species_id}\n")


def _collect_images_by_dominance(
    dataset_path: str,
    class_name: str,
    dominant_species: Optional[Dict[str, List[str]]],
    species_to_id: Dict[str, int],
    species_dict: Dict[int, str],
    image_list: List[Tuple[str, int]],
    current_id: int,
) -> int:
    """
    Collects image paths for dominant and non-dominant species from the dataset.

    First, it processes dominant species, assigning unique species IDs. Then, it 
    processes non-dominant species, assigning them to the "Other" category.

    Args:
        dataset_path: The path to the dataset containing species folders.
        class_name: The species class to process.
        dominant_species: A dictionary of dominant species by class.
        species_to_id: A mapping of species names to unique IDs.
        species_dict: A mapping of species IDs to species names.
        image_list: The list to accumulate image paths and their corresponding species IDs.
        current_id: The current species ID to assign.

    Returns:
        int: The updated species ID after processing the species.

    Raises:
        FailedOperation: If no dominant species are found for the given class.
    """
    dominant_set: Optional[Set[str]] = set(dominant_species.get(class_name, [])) if dominant_species else None

    if dominant_set is None:
        raise FailedOperation("No dominance species detected, double check the dataset.")

    # First pass: dominant species
    for species in sorted(os.listdir(dataset_path)):
        species_path = os.path.join(dataset_path, species)
        if not os.path.isdir(species_path):
            continue
        if species in dominant_set:
            label = species_to_id.setdefault(species, current_id)
            if label == current_id:
                species_dict[current_id] = species
                current_id += 1
            for img_file in os.listdir(species_path):
                img_path = os.path.join(species_path, img_file)
                image_list.append((img_path, label))

    # Second pass: non-dominant species → "Other"
    other_label = sum(len(species_list) for species_list in dominant_species.values())  # type: ignore
    for species in os.listdir(dataset_path):
        species_path = os.path.join(dataset_path, species)
        if not os.path.isdir(species_path):
            continue
        if species not in dominant_set:
            for img_file in os.listdir(species_path):
                img_path = os.path.join(species_path, img_file)
                image_list.append((img_path, other_label))

    if "Other" not in species_dict.values():
        species_dict[other_label] = "Other"

    return current_id


def _write_species_composition(output_path: str, image_list: List[Tuple[str, int]], species_dict: Dict[int, str]):
    species_composition: Dict[str, int] = {}
    for species_label, species_name in species_dict.items():
        current_species = [1 for label in image_list if label[1] == species_label]
        total_species = sum(current_species)
        species_composition[species_name] = total_species
    write_data_to_json(output_path, "species_composition", species_composition)

def _write_species_lists(
    base_output_path: str,
    image_list: List[Tuple[str, int]],
    species_dict: Dict[int, str],
):
    """
    Writes species-specific image lists to the specified output path.

    Groups the images by species, creates directories for each species, and saves 
    the list of image paths to a Parquet file for each species.

    Args:
        base_output_path (str): The base directory where species lists will be saved.
        image_list (List[Tuple[str, int]]): A list of image paths and their corresponding species IDs.
        species_dict (Dict[int, str]): A dictionary mapping species IDs to species names.
    """

    species_list_dir = os.path.join(base_output_path, "species_lists")
    os.makedirs(species_list_dir, exist_ok=True)

    species_group: Dict[str, List[Tuple[str, int]]] = {}
    for img_path, label in image_list:
        species = species_dict[label]
        if species not in species_group:
            species_group[species] = []
        species_group[species].append((img_path, label))


    for species, tuple_list in tqdm(species_group.items(), f"Writing species specific manifest to {species_list_dir}"):
        class_name = tuple_list[0][0].split(os.sep)[-3]
        species_dir = os.path.join(species_list_dir, class_name, species)
        os.makedirs(species_dir, exist_ok=True)
        file_name = os.path.join(species_dir, "images.parquet")

        save_manifest_parquet(tuple_list, file_name)

        # with open(os.path.join(species_dir, "images.txt"), "w") as file:
        #     file.write("\n".join(tuple_list))


def run_manifest_generator(
    data_dir: str,
    output_dir: str,
    dataset_properties_path: str,
    train_size: float,
    random_state: int,
    target_classes: List[str],
    threshold: float
):
    """
    Generates a dataset manifest of image paths and labels. Splits the dataset 
    into training and validation sets based on the dominant threshold and saves
    species data to Parquet and JSON files.

    Args:
        data_dir: The directory containing species class folders.
        output_dir: The directory to save the output manifests and species data.
        dataset_properties_path: Path to the properties file containing dataset information.
        train_size: The proportion of the dataset to use for training.
        random_state: Random seed for reproducibility of train/test split.
        target_classes: List of species classes to include in the manifest.
        threshold: The threshold for identifying dominant species based on image count.

    Raises:
        FailedOperation: If there are issues processing the dataset or saving the manifest.
    """
    os.makedirs(output_dir, exist_ok=True)

    dominant_species = _identifying_dominant_species(dataset_properties_path, threshold, target_classes)

    species_to_id: Dict[str, int] = {}
    species_dict: Dict[int, str] = {}
    image_list: List[Tuple[str, int]] = []
    current_id = 0

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path) or class_name == "species_lists":
            continue

        current_id = _collect_images_by_dominance(
            class_path,
            class_name,
            dominant_species,
            species_to_id,
            species_dict,
            image_list,
            current_id,
        )
    species_dict = dict(sorted(species_dict.items()))

    save_manifest_parquet(image_list, os.path.join(output_dir, "dataset_manifest.parquet"))

    with open(os.path.join(output_dir, "dominant_labels.json"), "w", encoding="utf-8") as file:
        json.dump(species_dict, file, indent=4)
    
    _write_species_composition(os.path.join(output_dir, "species_composition.json"), image_list, species_dict)

    train_data, val_data = train_test_split(
        image_list,
        train_size=train_size,
        random_state=random_state,
        stratify=[label for _, label in image_list],
    )

    with open(
        os.path.join(output_dir, "dataset_species_labels.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(species_dict, file, indent=4)

    save_manifest_parquet(train_data, os.path.join(output_dir, "train.parquet"))
    save_manifest_parquet(val_data, os.path.join(output_dir, "val.parquet"))

    _write_species_lists(output_dir, image_list, species_dict)

    print(f"Dominant manifest created in: {output_dir}")
    print(f"Total species (with 'Other'): {len(species_dict)}")
    print(f"Total Images: {len(image_list)} | Train: {len(train_data)} | Val: {len(val_data)}")
