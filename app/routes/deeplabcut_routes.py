import os
import mimetypes
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from services.deeplabcut_service import process_video_with_params


router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Hello World"}


@router.post("/analyze-video")
async def analyze_video(file: UploadFile):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        output_path = f"temp_output_{file.filename}"
        keypoints_list = process_video_with_params(temp_file_path, output_path)

        os.remove(temp_file_path)
        os.remove(output_path)

        return {"keypoints": keypoints_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/{filename}")
async def stream_video(filename: str):
    video_path = Path(filename)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    mime_type, _ = mimetypes.guess_type(video_path)

    return StreamingResponse(video_path.open("rb"), media_type=mime_type)


__all__ = ["router"]
