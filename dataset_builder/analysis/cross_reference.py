import os
from typing import List

from dataset_builder.analysis.matching import (
    cross_reference_set,
)
from dataset_builder.core.exceptions import FailedOperation
from dataset_builder.core.log import log
from dataset_builder.core.utility import (
    _is_json_file,
    read_species_from_json,
    write_data_to_json,
)


def run_cross_reference(
    output_file_path: str,
    json_1_path: str,
    json_2_path: str,
    data_1_name: str,
    data_2_name: str,
    target_classes: List[str],
    verbose: bool = False,
    overwrite: bool = False,
):
    """
    Performs a cross-reference between two species datasets to identify 
    matching species based on the provided target classes and outputs the 
    results to a JSON file.

    Args:
        output_file_path (str): The file path where the results will be saved.
        json_1_path (str): The file path to the first species dataset in JSON format.
        json_2_path (str): The file path to the second species dataset in JSON format.
        data_1_name (str): A name or label for the first dataset (e.g., dataset name).
        data_2_name (str): A name or label for the second dataset (e.g., dataset name).
        target_classes (List[str]): A list of species classes to use for the cross-reference.
        verbose (bool, optional): A flag to print additional details about the matching process. Defaults to False.
        overwrite (bool, optional): A flag to determine whether to overwrite an existing output file. Defaults to False.

    Raises:
        FailedOperation: If one or both datasets are empty or if any dataset is not a valid JSON file.

    Returns:
        None: This function does not return any value but writes the cross-referenced species data to the specified output file.

    Prints:
        - A message if the output file already exists and `overwrite` is False.
        - The total number of matches found between the datasets if `verbose` is True.
    """
    display_name = f"Matched species between {data_1_name} and {data_2_name}"

    if os.path.isfile(output_file_path) and not overwrite:
        print(f"{output_file_path} already exists, skipping web crawl.")
        return

    if not _is_json_file(json_1_path) or not _is_json_file(json_2_path):
        raise FailedOperation("JSON file not found.")

    dataset_1 = read_species_from_json(json_1_path)
    dataset_2 = read_species_from_json(json_2_path)

    if not dataset_1 or not dataset_2:
        raise FailedOperation(
            "One or both species dataset are empty. Cross-reference aborted."
        )

    match_species, total_matches = cross_reference_set(
        dataset_1, dataset_2, target_classes
    )

    log(f"Total matches: {total_matches}", verbose)

    write_data_to_json(output_file_path, display_name, match_species)
