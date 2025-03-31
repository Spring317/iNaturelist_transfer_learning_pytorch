import os
import shutil
from tqdm import tqdm
from typing import List
from dataset_builder.core.log import log
from dataset_builder.core.utility import SpeciesDict, read_species_from_json, _is_species_dict
from dataset_builder.core.exceptions import FailedOperation

def run_copy_matched_species(
    src_dataset: str,
    dst_dataset: str,
    matched_species_json: str,
    target_classes: List[str],
    verbose: bool = False,
) -> None:
    """
    Copies species data from a source dataset to a destination dataset based on 
    a matched species JSON file. Only species that are present in the matched 
    species JSON and belong to the specified target classes will be copied.

    Args:
        src_dataset: Path to the source dataset directory containing species data.
        dst_dataset: Path to the destination dataset directory where species data will be copied.
        matched_species_json: Path to the JSON file containing matched species information.
        target_classes: List of species classes to be copied from the matched species.
        verbose: Whether to print detailed information about the copying process. Defaults to False.

    Raises:
        FailedOperation: If the matched species JSON file cannot be found, is in an invalid format, 
                         or if not all species are successfully copied.
    """
    if not os.path.isfile(matched_species_json):
        raise FailedOperation("Cannot find matched species JSON. Cannot proceed with copying matched species")

    matched_species: SpeciesDict = read_species_from_json(matched_species_json)
    if not _is_species_dict(matched_species):
        raise FailedOperation("Invalid matched species JSON format, please check the file.")
    

    species_num = sum(len(species_list) for species_list in matched_species.values())
    species_copied = 0

    print(f"Copying data to {dst_dataset}")
    for class_name, species_set in matched_species.items():
        if class_name not in target_classes:
            continue

        species_iter = species_set if verbose else tqdm(species_set, f"Copy data in {class_name}")

        for species_name in species_iter:
            src_dir = os.path.join(src_dataset, class_name, species_name)
            dst_dir = os.path.join(dst_dataset, class_name, species_name)

            if os.path.exists(src_dir):
                species_copied += 1
                os.makedirs(dst_dir, exist_ok=True)

                for item in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, item)
                    dst_file = os.path.join(dst_dir, item)
                    if os.path.isfile(src_file):
                        if not os.path.isfile(dst_file):
                            shutil.copy2(src_file, dst_dir)
                            log(f"{species_copied}/{species_num} Copied: {class_name}/{species_name}/{item}", verbose)
                        else:
                            log(f"{species_copied}/{species_num} File exists - skipping: {class_name}/{species_name}/{item}", verbose)
                            pass
            else:
                log(f"{species_copied}/{species_num} Missing source directory: {src_dir}", True, "ERROR")

    if (species_copied != species_num):
        raise FailedOperation(f"Failed to copy all matches species copied {species_copied}/{species_num}")
