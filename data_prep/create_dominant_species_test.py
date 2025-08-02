import os
import json 

def create_special_test(path: str):
    """
    Create a special test for the dominant species function.
    arguments:
        path (str): The path to the dominant class (contain all the images paths)
    """
    images = os.listdir(path)

    # select 1000 random images from the list and save it to a dict
    for i in range(len(images))
        j = random.randint(0, len(images) - 1)
    
    # Create a JSON file with the dominant species
    print(f"Special test created at {path}")
    
