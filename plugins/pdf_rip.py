import os
import logging
from pdfrw import PdfReader, PdfWriter
from starlette.responses import FileResponse
from modules.plugins import PluginBase
from utils.file_utils import delete_file, random_filename
from fastapi import BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel


class PDFRipRequest(BaseModel):
    pdf: UploadFile
    pages: str


@PluginBase.router.post("/pdf/rip", tags=["PDF"])
async def pdf_rip(background_tasks: BackgroundTasks, req: PDFRipRequest = Depends()):

    new_pdf_path = random_filename("pdf")

    try:
        # support both comma-separated numbers or ranges in combination
        pages = []
        for page in req.pages.split(","):
            if "-" in page:
                start, end = page.split("-")
                pages.extend(range(int(start) - 1, int(end)))
            else:
                pages.append(int(page) - 1)

        new_pdf: PdfWriter = PdfWriter(new_pdf_path)

        pdf = PdfReader(req.pdf.file)
        for page in pages:
            new_pdf.addPage(pdf.getPage(page))

        new_pdf.write()

        return FileResponse(
            new_pdf_path,
            media_type="application/pdf",
            filename=os.path.basename(new_pdf_path),
        )

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(new_pdf_path):
            background_tasks.add_task(delete_file, new_pdf_path)
