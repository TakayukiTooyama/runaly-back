from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import HTTPException
from models.deeplabcut_model import dlc_live


def bbox_to_tuple(dict_bbox: Dict[str, int]) -> Tuple[int]:
    return (dict_bbox["x"], dict_bbox["y"], dict_bbox["width"], dict_bbox["height"])


def extract_keypoints(
    video_path: Path, human_bbox: Dict[str, int]
) -> List[List[List[float]]]:
    """ビデオからキーポイントを抽出する

    Args:
        video_path (Path): _description_
        video_time (Dict[str, float]): _description_
        human_bbox (Dict[str, int]): _description_

    Raises:
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    tracker = cv2.TrackerCSRT().create()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to read video file")

    # 出力ビデオの設定
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(
        "./public/video/output_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (frame_width, frame_height),
    )

    keypoints = []
    ret, frame = cap.read()
    if not ret:
        raise HTTPException(
            status_code=400, detail="Error reading the video or video has no frames."
        )
    tracker.init(frame, bbox_to_tuple(human_bbox))
    dlc_live.init_inference(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_index = 0

    while ret:
        # フレームでオブジェクトを追跡
        success, bbox = tracker.update(frame)
        if success:
            # トラッキング成功: バウンディングボックスを描画
            (x, y, width, height) = [int(v) for v in bbox]
            # 拡張
            x = x - 80
            y = y - 80
            width = width + 160
            height = height + 160
            # クロップ
            cropped_frame = frame[y : y + height, x : x + width]
            frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pose = dlc_live.get_pose(frame_rgb)
            keypoint = np.floor(pose[:, :2]).tolist()
            adjusted_keypoints = [(kp[0] + x, kp[1] + y) for kp in keypoint]
            keypoints.append(adjusted_keypoints)
            # バウンディングボックスとキーポイントを描画
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            for kpt in adjusted_keypoints:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (0, 255, 0), -1)

            # 画像としてフレームを保存
            cv2.imwrite(f"./public/image/wakui/frame_{frame_index}.png", frame)
            frame_index += 1
            out.write(frame)
            # print(keypoints)
            # confidences = pose[:, 2]
            # for kpt, conf in zip(keypoints, confidences):
            #     if conf > 0.5:
            #         cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (0, 0, 255), -1)
        else:
            print("トラッキング失敗")

        ret, frame = cap.read()

    cap.release()

    return keypoints
