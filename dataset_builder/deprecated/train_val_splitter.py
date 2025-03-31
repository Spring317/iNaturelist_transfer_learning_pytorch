import os
import json

from sklearn.model_selection import train_test_split
from typing import Dict, Tuple


def save_data_manifest(file_path, data):
    """
    Write extracted data to a file line by line.
    """
    with open(file_path, "w") as file:
        for img_path, species_id in data:
            file.write(f"{img_path}: {species_id}\n")


def collect_images_from_class(
    dataset_path: str,
    class_name: str,
    included_classes: list[str],
    species_to_id: Dict[str, int],
    species_dict: Dict[int, str],
    image_list: list[Tuple[str, int]],
    current_id: int,
    is_other: bool = False,
) -> int:
    """
    Collects image paths and assigns label IDs for each species within a given class folder.
    This function only processes the path if it's in `included_classes` or is named "Other"

    This function updates:
    - species_to_id: mapping from species name to label ID
    - species_dict: mapping from label ID to species name
    - image_list: list of (image_path, label_id) pairs

    Args:
        dataset_path: path to the directory containing species folders.
        class_name: name of the taxonomic class (e.g., "Aves", "Insecta").
        included_classes: classes to include in the dataset manifest.
        species_to_id: dictionary mapping species name to label ID.
        species_dict: dictionary mapping label ID to species name.
        image_list: accumulates (image_path, label_id) pairs.
        current_id: the next available id to assign.
        isOther: whether this class is the special "Other" category

    Returns:
        int: updated `current_id` after assigning new species
    """
    if not is_other and class_name not in included_classes:
        return current_id

    for species in os.listdir(dataset_path):
        species_path = os.path.join(dataset_path, species)
        if not os.path.isdir(species_path):
            continue

        if is_other:
            label = current_id
        else:
            if species in species_to_id:
                label = species_to_id[species]
            else:
                label = current_id
                species_to_id[species] = current_id
                species_dict[current_id] = species
                current_id += 1

        for img_file in os.listdir(species_path):
            img_path = os.path.join(species_path, img_file)
            image_list.append((img_path, label))

    if is_other:
        species_dict[current_id] = "Other"
        current_id += 1

    return current_id


def write_species_lists(
    base_output_path: str,
    image_list: list[Tuple[str, int]],
    species_dict: Dict[int, str],
):
    """
    Write a manifest for each species, organized in subfolders by class and species name.

    Args:
        output_root: base output directory for storing species lists.
        image_list: list of (image_path, label) pairs.
        species_dict: Mapping from label ID to species name.
    """
    species_list_dir = os.path.join(base_output_path, "species_lists")
    os.makedirs(species_list_dir, exist_ok=True)

    species_group: Dict[str, list[str]] = {}
    for img_path, label in image_list:
        species = species_dict[label]
        if species not in species_group:
            species_group[species] = []
        species_group[species].append(f"{img_path}: {label}")

    for species, lines in species_group.items():
        # <class>/<species>/<images>
        # class_name = os.path.basename(
        #     os.path.dirname(os.path.dirname(lines[0].split(":")[0]))
        # )
        path_parts = lines[0].split(os.sep)
        class_name = path_parts[-3]
        species_dir = os.path.join(species_list_dir, class_name, species)
        os.makedirs(species_dir, exist_ok=True)

        with open(os.path.join(species_dir, "images.txt"), "w") as file:
            file.write("\n".join(lines))


def split_dataset(
    data_dir: str,
    output_dir: str,
    included_classes: list[str],
    train_size: float,
    random_state: int,
):
    os.makedirs(output_dir, exist_ok=True)

    species_to_id: Dict[str, int] = {}
    species_dict: Dict[int, str] = {}
    image_list: list[Tuple[str, int]] = []
    species_id_counter = 0

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        if class_name == "species_lists" and class_name == "Other":
            continue

        species_id_counter = collect_images_from_class(
            class_path,
            class_name,
            included_classes,
            species_to_id,
            species_dict,
            image_list,
            species_id_counter,
        )

    class_path = os.path.join(data_dir, "Other")
    if os.path.isdir(class_path):
        species_id_counter = collect_images_from_class(
            class_path,
            "Other",
            included_classes,
            species_to_id,
            species_dict,
            image_list,
            species_id_counter,
            is_other=True,
        )

    save_data_manifest(os.path.join(data_dir, "dataset_manifest.txt"), image_list)

    with open(
        os.path.join(data_dir, "dataset_species_labels.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(species_dict, file, indent=4)

    train_data, val_data = train_test_split(
        image_list,
        train_size=train_size,
        random_state=random_state,
        stratify=[label for _, label in image_list],
    )

    save_data_manifest(os.path.join(data_dir, "train.txt"), train_data)
    save_data_manifest(os.path.join(data_dir, "val.txt"), val_data)

    write_species_lists(data_dir, image_list, species_dict)

    print(f"train.txt and val.txt created in {data_dir}")
    print(f"Total species: {species_id_counter}")
    print(
        f"Total Images: {len(image_list)} | Train: {len(train_data)} | Val: {len(val_data)}"
    )


def run_split(config):
    data_dirs = [
        config["paths"]["src_dataset"],
        config["paths"]["dst_dataset"],
        config["paths"]["inter_dataset"],
        config["paths"]["dst_dataset_small"],
    ]
    data_dir = config["paths"]["dst_dataset"]
    output_dir = config["paths"]["output_dir"]
    included_classes = config["train_val_split"]["included_classes"]
    train_size = config["train_val_split"]["train_size"]
    random_state = config["train_val_split"]["random_state"]
    for data_dir in data_dirs:
        split_dataset(data_dir, output_dir, included_classes, train_size, random_state)
