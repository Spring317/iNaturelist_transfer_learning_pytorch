from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from dataset_builder.core.utility import _prepare_data_cdf_ppf


def _identifying_dominant_species(properties_json_path: str, threshold: float, classes_to_analyze: List[str]) -> Optional[Dict[str, List[str]]]:
    """
    Identifies dominant species in the given classes based on a specified image count threshold.

    The function calculates the cumulative distribution of image counts for each species class,
    and identifies species whose image counts exceed the threshold defined by the given percentile.

    Args:
        properties_json_path: Path to the JSON file containing species image data.
        threshold: The cumulative percentage threshold (e.g., 0.5 for 50%).
        classes_to_analyze: List of species classes to analyze.

    Returns:
        Dict(str, List[str]): A dictionary where keys are species class names, and values are lists of dominant species names.
        Returns None if the data preparation fails for any class.
    """
    species_data: Dict[str, list[str]] = defaultdict(list)
    for species_class in classes_to_analyze:
        result = _prepare_data_cdf_ppf(properties_json_path, species_class)
        if result is None:
            print(f"ERROR: Data preparation failed for {species_class}")
            return None
        species_names, sorted_image_counts = result
        total_images = sum(sorted_image_counts)
        cumulative_images = np.cumsum(sorted_image_counts) 
        cdf_values = cumulative_images / total_images
        sorted_images = np.array(sorted_image_counts)
        filtered_index = np.argmax(cdf_values >= threshold)
        thresholded_image_count = sorted_images[filtered_index]

        dominant_species = [species for species, count in zip(species_names, sorted_image_counts) if count >= thresholded_image_count]
        species_data[species_class] = dominant_species
    return species_data
