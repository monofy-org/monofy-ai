import os

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