import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml  # type: ignore
from tqdm import tqdm  # type: ignore

from dataset_builder.core.log import log

SpeciesDict = Dict[str, List[str]]


def banner(title: str):
    """
    Prints a banner with a title, surrounded by '#' characters.

    Args:
        title (str): The title to display in the banner.
    """
    line = "#" * 60
    print(f"{line}\n{title.upper()}\n{line}")


def _is_json_file(json_path: str) -> bool:
    """
    Checks if the given file path points to a valid JSON file.

    Args:
        json_path (str): The file path to check.

    Returns:
        bool: True if the file is a JSON file, False otherwise.
    """
    if os.path.isfile(json_path) and os.path.basename(json_path).lower().endswith(
        ".json"
    ):
        return True
    return False


def _is_species_dict(obj: Any) -> bool:
    """
    Checks if the given object is a valid species dictionary.

    The object must be a dictionary where keys are strings (species classes) 
    and values are lists of species names (strings).

    Args:
        obj (Any): The object to check.

    Returns:
        bool: True if the object is a valid species dictionary, False otherwise.
    """
    if not isinstance(obj, dict):
        return False
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, list):
            return False
        if not all(isinstance(species, str) for species in v):
            return False
    return True


def save_manifest_parquet(manifest: List[Tuple[str, int]], path: str):
    """
    Saves a dataset manifest to Parquet format.

    Args:
        manifest: List of (image_path, label_id)
        path: Output file path (.parquet)
    """
    df = pd.DataFrame(manifest, columns=["image_path", "label_id"])
    df.to_parquet(path, index=False)


def load_manifest_parquet(path: str) -> List[Tuple[str, int]]:
    """
    Loads a Parquet dataset manifest and returns it as a list of tuples.
    """
    df = pd.read_parquet(path)
    return list(df.itertuples(index=False, name=None))


def write_data_to_json(file_output_path: str, display_name: str, species_data) -> None:
    """
    Writes data to a JSON file.

    Args:
        file_output_path: The path where the JSON file will be saved.
        species_data: Dictionary like data structure

    Raises:
        IOError: If an error occur while writing to the file.
    """
    try:
        os.makedirs(os.path.dirname(file_output_path), exist_ok=True)

        with open(file_output_path, "w", encoding="utf-8") as f:
            json.dump(species_data, f, indent=4)

        print(f"{display_name} → {file_output_path}")
    except IOError as e:
        log(f"Error writing to file {file_output_path}: {e}", True, "ERROR")


def read_species_from_json(file_input_path: str) -> SpeciesDict:
    """
    Reads species and subspecies data from a JSON file.

    Args:
        file_output_path: The path to the JSON file.

    Returns:
        SpeciesDict (Dict[str, List[str]]): Dictionary containing species as keys and their species as values.

    Raises:
        IOError: If the file cannot be read or does not exist.
        JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(file_input_path, "r", encoding="utf-8") as f:
            species_data = json.load(f)

            print(f"Successfully loaded species data from {file_input_path}")
            return species_data

    except IOError as e:
        log(f"Error reading {file_input_path}: {e}", True, "ERROR")
        return {}

    except json.JSONDecodeError as e:
        log(f"Invalid JSON format in {file_input_path}: {e}", True, "ERROR")
        return {}


def _prepare_data_cdf_ppf(
    properties_json_path: str, class_to_analyze: str
) -> Optional[Tuple[List[str], List[int]]]:
    """Loads species image data from JSON dataset properties file, sorts by image count, and prepares data for CDF/PPF calculations.

    Args:
        properties_json_path: Path to the dataset properties JSON file.
        class_to_analyze: The target class (e.g., 'Aves', 'Insecta').

    Returns:
        Optional[Tuple[List[str], List[int]]]: Species names and corresponding image counts sorted by number of images.
    """
    try:
        with open(properties_json_path, "r", encoding="utf-8") as file:
            species_data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {properties_json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Not a valid JSON file: {e}")
        return None

    species_images_data: Dict[str, int] = species_data.get(class_to_analyze, {})
    if not species_images_data:
        print(f"ERROR: Class '{class_to_analyze}' not found or contains no data")
        return None
    sorted_species = sorted(
        species_images_data.items(), key=lambda x: x[1], reverse=True
    )
    species_names, image_counts = zip(*sorted_species)

    return list(species_names), list(image_counts)


def cleanup(**remove_paths):
    """
    Removes the specified directories and their contents.

    Args:
        **remove_paths: One or more paths to directories that should be removed.
    """
    for path in remove_paths:
        shutil.rmtree(path)
