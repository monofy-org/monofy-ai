import json
import logging

from fastapi.responses import JSONResponse
from modules import plugins


def zip_to_coordinates(zip_code) -> tuple[float, float]:
    zip_string = zip_code if isinstance(zip_code, str) else str(zip_code)
    starting_digit = zip_string[0]
    file = f"submodules/zip-code-json/zip{starting_digit}.json"
    try:
        with open(file, "r") as f:
            content = f.read()
            json_content = content.lstrip("zips=")
            data: dict = json.loads(json_content)
            coords = data.get(zip_string)
            if coords is None:
                return None, None
            return coords[0], coords[1]
    except Exception as e:
        logging.error(e, exc_info=True)
        return None, None


@plugins.router.get("/geo/zip2coords")
async def zip_to_coordinates_get(zip: str):
    # Sanitize the input to prevent path traversal
    sanitized_zip = "".join(c for c in zip if c.isalnum())
    latlong = zip_to_coordinates(sanitized_zip)

    return JSONResponse({"longitude": latlong[0], "latitude": latlong[1] })
