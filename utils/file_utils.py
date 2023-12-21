import os
import random
import string


def ensure_folder_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder {path}")


def delete_file(file_path: str):
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")


def random_filename(file_extension: str = None, length: int = 10):
    filename = "".join(random.choice(string.ascii_letters) for _ in range(length))
    if file_extension is not None:
        filename += f".{file_extension}"
    return filename
