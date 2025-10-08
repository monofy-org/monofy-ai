from typing import List, Optional

from fastapi.responses import JSONResponse
from numpy import ndarray
from modules.plugins import PluginBase, release_plugin, use_plugin
from pydantic import BaseModel
from utils.image_utils import get_image_from_request
from PIL import Image


class VisionFaceRecognitionRequest(BaseModel):
    target: str
    faces: str | List[str]
    folder: Optional[str] = None


class VisionFaceRecognitionResponse(BaseModel):
    scores: list[float]


class VisionFaceRecognitionPlugin(PluginBase):
    name = "vision_face_recognition"
    description = "Face recognition and similarity scoring"

    def get_similarity(
        self, request: VisionFaceRecognitionRequest
    ) -> VisionFaceRecognitionResponse:
        import face_recognition
        import numpy as np

        def load_image(file_path):
            image: Image.Image = get_image_from_request(file_path)
            encodings = face_recognition.face_encodings(np.array(image))
            if len(encodings) == 0:
                raise ValueError(f"No faces found in the image: {file_path}")
            return encodings[0]

        target_encoding = load_image(request.target)

        if isinstance(request.faces, str):
            faces_list = [request.faces]
        else:
            faces_list = request.faces

        scores = []
        for face in faces_list:
            match_encoding = load_image(face)
            distance = np.linalg.norm(target_encoding - match_encoding)
            similarity = 1 / (1 + distance)  # Convert distance to similarity score
            scores.append(similarity)

        if request.folder:
            # get all images in the folder and compute similarity
            import os

            folder_path = os.path.join(os.getcwd(), "faces", request.folder)
            if not os.path.exists(folder_path):
                raise ValueError(f"Folder does not exist: {folder_path}")
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(folder_path, filename)
                    match_encoding = load_image(file_path)
                    distance = np.linalg.norm(target_encoding - match_encoding)
                    similarity = 1 / (1 + distance)
                    scores.append(similarity)

        return VisionFaceRecognitionResponse(scores=scores)


@PluginBase.router.post(
    "/vision/face_recognition", response_model=VisionFaceRecognitionResponse
)
async def vision_face_recognition(request: VisionFaceRecognitionRequest):
    plugin: VisionFaceRecognitionPlugin = None
    try:
        plugin = await use_plugin(VisionFaceRecognitionPlugin)
        return plugin.get_similarity(request)
    except Exception as e:
        raise ValueError(f"Error processing face recognition: {str(e)}")
    finally:
        release_plugin(VisionFaceRecognitionPlugin)
