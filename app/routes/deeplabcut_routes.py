import json
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from services.analyze_service import process_analyze
from services.deeplabcut_service import extract_keypoints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    data: str = Form(...),
):
    video_data = json.loads(data)
    human_bbox = video_data["humanBBox"]
    cali_markers = video_data["caliMarkers"]
    cali_width = video_data["caliWidth"]
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        contents = file.file.read()
        with open(temp.name, "wb") as f:
            f.write(contents)
        keypoints = extract_keypoints(temp.name, human_bbox)
        analyzes = process_analyze(keypoints, cali_markers, cali_width)
        return JSONResponse(analyzes)
    except HTTPException as http_err:
        raise http_err
    except Exception as err:
        logger.error(f"Unexpected error: {err}")
        raise HTTPException(status_code=500, detail="Error processing the video")
    finally:
        file.file.close()
        os.remove(temp.name)


__all__ = ["router"]
