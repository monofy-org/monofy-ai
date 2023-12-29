from utils.file_utils import fetch_pretrained_model

MODEL_NAME = "Intel/dpt-large"

friendly_name = "dpt"
model = None
current_model_name: str = None
model_path: str = None


def load_model(model_name=MODEL_NAME):
    global current_model_name
    global model_path
    global model    
    
    path = fetch_pretrained_model(model_name, "dpt")

    if model is None:
        model = None  # TODO

        model = model
        model_name = model_name


def offload(for_task):
    # TODO
    pass


def generate():
    # TODO
    pass
