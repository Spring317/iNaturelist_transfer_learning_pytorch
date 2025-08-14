import os
import shutil
from typing import List, Tuple

from pipeline.utility import (
    manifest_generator_wrapper,
)


def create_DI() -> List[Tuple[str, int]]:
    _, _, global_image_data, global_species_labels, global_species_composition = (
        manifest_generator_wrapper(1.0)
    )
    return global_image_data


data = create_DI()

# Copy images listed in data to another folder. Following the structure of the original dataset
os.makedirs("test", exist_ok=True)
for image_path, species_id in data:
    class_path = f"{image_path.split('/')[-2]}"
    target_dir = f"test/{class_path}"

    print(f"Writing on {target_dir}")

    if not os.path.exists(target_dir):
        # Create the target_dirrget directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

    # Copy the image to the target directory
    shutil.copy(image_path, target_dir)
