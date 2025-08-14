import os


def create_testset(path: str):
    test_set = []
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):  # Ensure it's a directory
            for image in os.listdir(dir_path):
                img_path = os.path.join(dir_path, image)
                test_set.append((img_path, 0))
    return test_set
