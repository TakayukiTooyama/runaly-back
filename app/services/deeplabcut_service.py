import cv2
from fastapi import HTTPException

from models.deeplabcut_model import dlc_live
import numpy as np


def get_fourcc_from_filename(filename: str) -> str:
    ext = filename.split(".")[-1].lower()
    if ext == "mp4":
        return "mp4v"
    elif ext in ["avi", "mov"]:
        return "XVID"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def process_video_with_params(video_path, output_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to read video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        raise HTTPException(
            status_code=400, detail="Error reading the video or video has no frames."
        )

    dlc_live.init_inference(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    keypoints_list = []
    while ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ここでフレームをクロッピングして推定する

        pose = dlc_live.get_pose(frame_rgb)

        keypoints = np.floor(pose[:, :2]).tolist()
        print(keypoints)
        keypoints_list.append(keypoints)
        confidences = pose[:, 2]
        for kpt, conf in zip(keypoints, confidences):
            if conf > 0.5:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()

    return keypoints_list
