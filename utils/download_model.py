from huggingface_hub import hf_hub_download
import joblib
import os


def download_model(folder: str, model: str):
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Download and load the model
    model = joblib.load(
        hf_hub_download(repo_id=model, filename="config.json", local_dir=folder)
    )

    return model
