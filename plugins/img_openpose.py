import logging
import os

import cv2
import numpy as np
import rembg
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse

from classes.requests import ImageProcessingRequest
from modules.plugins import PluginBase, use_plugin
from utils.file_utils import random_filename
from utils.image_utils import (
    get_image_from_request,
    image_to_base64_no_header,
)


@PluginBase.router.post("/img/openpose", tags=["Image Processing"])
async def estimate_pose(req: ImageProcessingRequest):
    try:
        from models.openpose.python.openpose_python import op

        if not os.path.exists("models/openpose"):
            raise Exception(
                "Please download OpenPose from https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases and copy the openpose folder into .\models"
            )

        image = get_image_from_request(req.image)

        temp_file = random_filename("png")
        image.save(temp_file, "PNG")

        
        opWrapper = op.WrapperPython()
        opWrapper.configure(dict())
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(temp_file)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)

        if req.return_json:            
            image = image.convert("RGBA")
            return {
                "images": [image_to_base64_no_header(image)],
            }

        return FileResponse(image)

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@PluginBase.router.get("/img/rembg", tags=["Image Processing"])
async def remove_background_from_url(req: ImageProcessingRequest = Depends()):
    return await estimate_pose(req)
