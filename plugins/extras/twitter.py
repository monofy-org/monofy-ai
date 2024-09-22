import logging
import os
import re
import bs4
from fastapi.responses import FileResponse
import requests
from modules.plugins import router
from utils.file_utils import random_filename

router.get("/twitter/download", tags=["Twitter Video Download"])


async def download_video(
    url: str,
):
    try:
        api_url = f"https://twitsave.com/info?url={url}"

        response = requests.get(api_url)
        data = bs4.BeautifulSoup(response.text, "html.parser")
        download_button = data.find_all("div", class_="origin-top-right")[0]
        quality_buttons = download_button.find_all("a")
        highest_quality_url = quality_buttons[0].get(
            "href"
        )  # Highest quality video url

        logging.info(f"Downloading video: {highest_quality_url}")

        file_name = (
            data.find_all("div", class_="leading-tight")[0]
            .find_all("p", class_="m-2")[0]
            .text
        )  # Video file name
        file_name = (
            re.sub(r"[^a-zA-Z0-9]+", " ", file_name).strip() + ".mp4"
        )  # Remove special characters from file name

        filename = random_filename("mp4")

        logging.info(f"Downloaded video: {highest_quality_url}")

        with open(filename, "wb") as f:
            f.write(requests.get(highest_quality_url).content)

        return FileResponse(filename, filename=os.path.basename(filename))
    except Exception as e:
        logging.error(e, exc_info=True)
    finally:
        os.remove(filename)
