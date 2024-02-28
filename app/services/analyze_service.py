import math
from typing import Dict, List

import numpy as np

BODY_PARTS = [
    "Head",
    "REar",
    "LEar",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "RToe",
    "RHeel",
    "LToe",
    "LHeel",
]


def convert_keypoints(
    keypoints: List[List[List[float]]], body_parts: List[str] = BODY_PARTS
) -> List[Dict[str, Dict[str, float]]]:
    """マジックナンバーを防ぐために、部位名や座標をキーとした辞書に変換する

    Args:
        keypoints (List[List[List[float]]]): DeepLabCutから出力されるフレームごとのキーポイント
        body_parts (List[str], optional): 部位のリスト. Defaults to BODY_PARTS.

    Returns:
        List[Dict[str, Dict[str, float]]]: 変換後のキーポイント
    """

    converted_keypoints = []
    for frame in keypoints:
        keypoints_dict = {
            part: {"x": coord[0], "y": coord[1]}
            for part, coord in zip(body_parts, frame)
        }
        converted_keypoints.append(keypoints_dict)

    return converted_keypoints


def convert_keypoints_to_dict(
    keypoints: List[List[List[float]]], body_parts: List[str] = BODY_PARTS
) -> Dict[str, List[List[float]]]:
    """DeepLabCutから出力されるフレームごとのキーポイントの座標を、キーポイントごとの座標に変換する

    Args:
        keypoints (List[List[List[float]]]): DeepLabCutから出力されるフレームごとのキーポイント
        body_parts (List[str], optional): 部位のリスト. Defaults to BODY_PARTS.

    Returns:
        Dict[str, List]: 変換後のキーポイント
    """
    # {'Head': [], 'REar': [], ...}
    converted_keypoints = {part: [] for part in body_parts}

    # frame: [[x1, y1], [x2, y2], ...]
    for frame in keypoints:
        # print(frame)
        # keypoint: [x, y]
        for i, keypoint in enumerate(frame):
            # body_parts[0]: "Head"
            converted_keypoints[body_parts[i]].append(keypoint)

    return converted_keypoints


def calculate_euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """2点間のユークリッド距離を計算する

    Args:
        point1 (List[float]): 始点の座標
        point2 (List[float]): 終点の座標

    Returns:
        float: 2点間のユークリッド距離
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calc_scale(cali_marker: Dict[str, List[int]], cali_width: float) -> float:
    """実長換算に使用するためのスケールの計算

    Args:
        cali_marker (Dict[str, List[int]]): キャリブレーションマーカーの座標
        cali_dimension (float): キャリブレーションマーカーの実長

    Returns:
        float: 実長スケール
    """
    left_marker = [
        (cali_marker["leftBottom"]["x"] + cali_marker["leftTop"]["x"]) / 2,
        (cali_marker["leftBottom"]["y"] + cali_marker["leftTop"]["y"]) / 2,
    ]
    right_marker = [
        (cali_marker["rightBottom"]["x"] + cali_marker["rightTop"]["x"]) / 2,
        (cali_marker["rightBottom"]["y"] + cali_marker["rightTop"]["y"]) / 2,
    ]
    scale = cali_width / calculate_euclidean_distance(right_marker, left_marker)
    return round(scale, 5)


def center_position(part1: List[float], part2: List[float]) -> List[float]:
    """関節の中心座標

    Args:
        part1 (List[float]): 関節1の座標（例:[x1, y1]）
        part2 (List[float]): 関節2の座標（例:[x2, y2]）

    Returns:
        List[float]: 関節の中心座標
    """
    return [(part1[0] + part2[0]) / 2, (part1[1] + part2[1]) / 2]


def calc_part_cog(
    distal: List[float], proximal: List[float], com_position_ratio: float
) -> List[float]:
    """部位ごとの重心位置を求める

    Args:
        distal (List[float]): 部位の遠位座標（例：[x1, y1]）
        proximal (List[float]): 部位の近位座標（例：[x2, y2]）
        com_position_ratio (float): 部位の重心位置の比率（例:0.821）

    Returns:
        List[float]: 重心の座標
    """
    # 部位の遠位から近位までのベクトルを計算
    vector = [proximal[0] - distal[0], proximal[1] - distal[1]]

    # ベクトルに比率を掛けて、整数の重心の位置を計算
    com_x = round(distal[0] + vector[0] * com_position_ratio)
    com_y = round(distal[1] + vector[1] * com_position_ratio)

    return [com_x, com_y]


def calc_cog(keypoints: Dict[str, List[List[float]]], frame: int) -> List[List[float]]:
    """身体重心を求める

    Args:
        keypoints (Dict[str, List[List[float]]]): 変換された全身のキーポイント座標
        frame (int): フレーム番号

    Returns:
        List[float]: 1フレームの重心座標
    """
    parts = [
        # 頭部の重心
        calc_part_cog(
            keypoints["Head"][frame],
            center_position(keypoints["REar"][frame], keypoints["LEar"][frame]),
            0.821,
        ),
        # 右上腕の重心
        calc_part_cog(keypoints["RShoulder"][frame], keypoints["RElbow"][frame], 0.529),
        # 左上腕の重心
        calc_part_cog(keypoints["LShoulder"][frame], keypoints["LElbow"][frame], 0.529),
        # 右前腕の重心
        calc_part_cog(keypoints["RElbow"][frame], keypoints["RWrist"][frame], 0.415),
        # 左前腕の重心
        calc_part_cog(keypoints["LElbow"][frame], keypoints["LWrist"][frame], 0.415),
        # 胴体の重心
        calc_part_cog(
            center_position(
                keypoints["RShoulder"][frame], keypoints["LShoulder"][frame]
            ),
            center_position(keypoints["RHip"][frame], keypoints["LHip"][frame]),
            0.493,
        ),
        # 右大腿の重心
        calc_part_cog(keypoints["RHip"][frame], keypoints["RKnee"][frame], 0.475),
        # 左大腿の重心
        calc_part_cog(keypoints["LHip"][frame], keypoints["LKnee"][frame], 0.475),
        # 右下腿の重心
        calc_part_cog(keypoints["RKnee"][frame], keypoints["RAnkle"][frame], 0.406),
        # 左下腿の重心
        calc_part_cog(keypoints["LKnee"][frame], keypoints["LAnkle"][frame], 0.406),
        # 右足の重心
        calc_part_cog(keypoints["RHeel"][frame], keypoints["RToe"][frame], 0.011),
        # 左足の重心
        calc_part_cog(keypoints["LHeel"][frame], keypoints["LToe"][frame], 0.011),
    ]

    total_x = sum(part[0] for part in parts)
    total_y = sum(part[1] for part in parts)
    num_parts = len(parts)

    average_x = round(total_x / num_parts)
    average_y = round(total_y / num_parts)

    return [average_x, average_y]


def scale_coordinate(
    coordinate: List[float], origin: List[float], scale: float
) -> List[float]:
    """座標をスケールし、原点を基準に変換

    Args:
        coordinate (List[float]): 実長換算するキーポイント座標
        origin (List[float]): 原点の座標
        scale (float): 座標のスケール

    Returns:
        List[float]: 実長換算後のキーポイント座標
    """
    return [
        (coordinate[0] - origin["x"]) * scale,
        (coordinate[1] - origin["y"]) * scale,
    ]


def calc_cog_velocity(
    cogs: List[List[float]],
    origin: List[float],
    frame_rate: int,
    scale: float,
) -> List[float]:
    """重心速度の計算

    Args:
        cogs (List[List[float]]): 全フレームの重心位置
        origin (List[float]): 原点の座標
        frame_rate (int): フレームレート
        x_scale (float): x座標のスケール
        y_scale (float): y座標のスケール

    Returns:
        List[float]: 重心速度（m/s）
    """
    # 重心座標をスケールし、原点を基準に変換
    scaled_cog = [scale_coordinate(point, origin, scale) for point in cogs]

    velocity = []

    # 最初のフレームの速度計算
    vx_first = (-3 * scaled_cog[0][0] + 4 * scaled_cog[1][0] - scaled_cog[2][0]) / (
        2 / frame_rate
    )
    vy_first = (-3 * scaled_cog[0][1] + 4 * scaled_cog[1][1] - scaled_cog[2][1]) / (
        2 / frame_rate
    )
    velocity.append(round(math.sqrt(vx_first**2 + vy_first**2), 2))

    # 中間のフレームの速度計算
    for i in range(1, len(scaled_cog) - 1):
        vx = (scaled_cog[i + 1][0] - scaled_cog[i - 1][0]) / (2 / frame_rate)
        vy = (scaled_cog[i + 1][1] - scaled_cog[i - 1][1]) / (2 / frame_rate)
        velocity.append(round(math.sqrt(vx**2 + vy**2), 2))

    # 最後のフレームの速度計算
    vx_last = (3 * scaled_cog[-1][0] - 4 * scaled_cog[-2][0] + scaled_cog[-3][0]) / (
        2 / frame_rate
    )
    vy_last = (3 * scaled_cog[-1][1] - 4 * scaled_cog[-2][1] + scaled_cog[-3][1]) / (
        2 / frame_rate
    )
    velocity.append(round(math.sqrt(vx_last**2 + vy_last**2), 2))

    return velocity


def calc_hip_flexion_angle(
    shoulder: List[float], hip: List[float], knee: List[float]
) -> float:
    """股関節屈曲角度の計算

    Args:
        shoulder (List[float]): 肩峰の座標
        hip (List[float]): 大転子の座標
        knee (List[float]): 膝関節の座標

    Returns:
        float: 股関節屈曲角度（°）
    """
    return calc_angle(shoulder, hip, knee)


def calc_trunk_angle(
    right_shoulder: List[float],
    left_shoulder: List[float],
    right_hip: List[float],
    left_hip: List[float],
) -> float:
    """体幹角度の計算

    Args:
        left_shoulder (List[float]): 左肩の座標
        right_shoulder (List[float]): 右肩の座標
        left_hip (List[float]): 左腰の座標
        right_hip (List[float]): 右腰の座標

    Returns:
        float: 体幹角度（°）
    """
    # 左肩と右肩の中点
    shoulder_midpoint = [
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    ]

    # 左腰と右腰の中点
    hip_midpoint = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]

    # 体幹角度の計算（中点間の角度）
    return calc_angle(
        shoulder_midpoint, hip_midpoint, [shoulder_midpoint[0], hip_midpoint[1]]
    )


def calc_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """3点の座標から角度を計算する

    Args:
        a (List[float, float]): 3点の上部の座標
        b (List[float, float]): 3点の中心の座標
        c (List[float, float]): 3点の下部の座標

    Returns:
        float: 屈曲角度（°）
    """
    if None in a or None in b or None in c:
        return None
    else:
        # ベクトル生成
        vector_ba = np.array(a) - np.array(b)  # bからaへのベクトル
        vector_bc = np.array(c) - np.array(b)  # bからcへのベクトル

        # ベクトルの内積
        dot_product = np.dot(vector_bc, vector_ba)

        # ベクトルの大きさ（ノルム）
        norm1 = np.linalg.norm(vector_bc)
        norm2 = np.linalg.norm(vector_ba)

        # 角度（ラジアン）を計算
        angle_radian = np.arccos(dot_product / (norm1 * norm2))

        # 既存のラジアンから度への変換関数
        angle_degrees = angle_radian * (180.0 / np.pi)

        # ベクトルの外積
        cross_product = np.cross(vector_bc, vector_ba)

        # 3点の角度が180°を超える場合
        if (cross_product < 0) and (dot_product < 0):
            return round(360 - angle_degrees, 1)
        # 3点の上部が水平線よりも下にある場合
        elif (cross_product < 0) and (dot_product > 0):
            return round(-angle_degrees, 1)
        # それ以外
        else:
            return round(angle_degrees, 1)


# def _calc_angle(
#     a: List[float], b: List[float], c: List[float], direction: str
# ) -> float:
#     """3点の座標からの屈曲角度の計算

#     Args:
#         a (List[float]): 3点の上部の座標
#         b (List[float]): 3点の中心の座標
#         c (List[float]): 3点の下部の座標
#         direction (str): 進行方向（L: 右から左, R: 左から右）

#     Returns:
#         float: 屈曲角度（°）

#     Warnings:
#         膝関節屈曲角度など、反対の角度の場合は反対になります。
#     """
#     if None in a or None in b or None in c:
#         return None

#     # 左から右に向かう時の屈曲角度
#     if direction == "R":
#         # 中心から上部へのベクトル
#         vector_ba = np.array(a) - np.array(b)
#         angle1 = np.arctan2(vector_ba[1], vector_ba[0])
#         angle1_degrees = angle1 * (180.0 / np.pi)
#         # 中心から下部へのベクトル
#         vector_bc = np.array(c) - np.array(b)
#         angle2 = np.arctan2(vector_bc[1], vector_bc[0])
#         angle2_degrees = angle2 * (180.0 / np.pi)
#         return round(angle2_degrees + (-angle1_degrees), 1)

#     # 右から左に向かう時の屈曲角度
#     else:
#         # 中心から上部へのベクトル
#         vector_ba = np.array(a) - np.array(b)
#         angle1 = np.arctan2(vector_ba[1], vector_ba[0])
#         angle1_degrees = -angle1 * (180.0 / np.pi)
#         # 中心から下部へのベクトル
#         vector_bc = np.array(c) - np.array(b)
#         angle2 = np.arctan2(vector_bc[1], vector_bc[0])
#         # 通常は-だがこの角度を次に引きたいため+にしている
#         angle2_degrees = angle2 * (180.0 / np.pi)
#         angle3 = 360 - angle2_degrees
#         final = angle3 - angle1_degrees
#         if final < 0:
#             final += 360
#         # クラウチング姿勢の股関節角度や体幹角度時に使用
#         elif final >= 270:
#             final -= 360

#         return round(final, 1)


def calc_step_length(
    right_heel: List[float], left_heel: List[float], scale: float
) -> float:
    """ステップ長の計算

    Description:
        ステップ長とは、片足の踵が地面に接地する点から、次に反対側の踵が地面に接地する点までの距離

    Args:
        left_heel (List[float]): 左踵の座標
        right_heel (List[float]): 右踵の座標
        scale (float): 実長スケール

    Returns:
        float: ストライド長（単位は座標系に依存）
    """
    # 左踵と右踵の座標間のユークリッド距離を計算
    distance = math.sqrt(
        ((right_heel[0] - left_heel[0]) * scale) ** 2
        + ((right_heel[1] - left_heel[1]) * scale) ** 2
    )
    return round(distance, 3)


def process_analyze(
    keypoints: List[List[List[float]]],
    cali_marker: Dict[str, List[int]],
    cali_width: float,
) -> Dict[str, List]:
    scale = calc_scale(cali_marker, cali_width)
    keypoints_to_dict = convert_keypoints_to_dict(keypoints)
    convert_keypoint = convert_keypoints(keypoints)

    # 結果を格納するリスト
    cogs = []
    trunk_angles = []
    right_hip_flexion_angles = []
    left_hip_flexion_angles = []
    step_lengths = []

    for frame in range(len(keypoints_to_dict["Head"])):
        # 重心位置
        cog = calc_cog(keypoints_to_dict, frame)
        cogs.append(cog)

        # 体幹角度
        trunk_angle = calc_trunk_angle(
            keypoints_to_dict["RShoulder"][frame],
            keypoints_to_dict["LShoulder"][frame],
            keypoints_to_dict["RHip"][frame],
            keypoints_to_dict["LHip"][frame],
        )
        trunk_angles.append(trunk_angle)

        # 股関節屈曲角度
        right_hip_flexion_angle = calc_hip_flexion_angle(
            keypoints_to_dict["RShoulder"][frame],
            keypoints_to_dict["RHip"][frame],
            keypoints_to_dict["RKnee"][frame],
        )
        left_hip_flexion_angle = calc_hip_flexion_angle(
            keypoints_to_dict["LShoulder"][frame],
            keypoints_to_dict["LHip"][frame],
            keypoints_to_dict["LKnee"][frame],
        )
        left_hip_flexion_angles.append(left_hip_flexion_angle)
        right_hip_flexion_angles.append(right_hip_flexion_angle)

        # ステップ長
        step_length = calc_step_length(
            keypoints_to_dict["RHeel"][frame],
            keypoints_to_dict["LHeel"][frame],
            scale,
        )
        step_lengths.append(step_length)

    # 重心速度
    cogVelocities = calc_cog_velocity(cogs, cali_marker["origin"], 60, scale)

    return {
        "keypoints": convert_keypoint,  # List[Dict[str, Dict[str, float]]]
        "cogs": cogs,  # List[List[float]]
        "cogVelocities": cogVelocities,  # List[float]
        "trunkAngles": trunk_angles,  # List[float]
        "rightHipFlexionAngles": right_hip_flexion_angles,  # List[float]
        "leftHipFlexionAngles": left_hip_flexion_angles,  # List[float]
        "stepLengths": step_lengths,  # List[float]
    }


# if __name__ == "__main__":
#     keypoints = KEYPOINTS
#     cali_marker = {
#         "leftTop": {"x": 236, "y": 684},
#         "rightTop": {"x": 1310, "y": 683},
#         "leftBottom": {"x": 236, "y": 735},
#         "rightBottom": {"x": 1310, "y": 738},
#         "origin": {"x": 1325, "y": 705},
#     }
#     cali_width = 70
#     scale = calc_scale(cali_marker, cali_width)
#     keypoints_to_dict = convert_keypoints_to_dict(keypoints)
#     convert_keypoint = convert_keypoints(keypoints)

#     # # 結果を格納するリスト
#     cogs = []
#     trunk_angles = []
#     right_hip_flexion_angles = []
#     left_hip_flexion_angles = []
#     step_lengths = []

#     for frame in range(len(keypoints_to_dict["Head"])):
#         # 重心位置
#         cog = calc_cog(keypoints_to_dict, frame)
#         cogs.append(cog)

#         # 体幹角度
#         trunk_angle = calc_trunk_angle(
#             keypoints_to_dict["RShoulder"][frame],
#             keypoints_to_dict["LShoulder"][frame],
#             keypoints_to_dict["RHip"][frame],
#             keypoints_to_dict["LHip"][frame],
#         )
#         trunk_angles.append(trunk_angle)

#         # 股関節屈曲角度
#         right_hip_flexion_angle = calc_hip_flexion_angle(
#             keypoints_to_dict["RShoulder"][frame],
#             keypoints_to_dict["RHip"][frame],
#             keypoints_to_dict["RKnee"][frame],
#         )
#         left_hip_flexion_angle = calc_hip_flexion_angle(
#             keypoints_to_dict["LShoulder"][frame],
#             keypoints_to_dict["LHip"][frame],
#             keypoints_to_dict["LKnee"][frame],
#         )
#         left_hip_flexion_angles.append(left_hip_flexion_angle)

#         # ステップ長
#         step_length = calc_step_length(
#             # keypoints_to_dict["RHeel"][frame],
#             # keypoints_to_dict["LHeel"][frame],
#             keypoints_to_dict["RAnkle"][frame],
#             keypoints_to_dict["LAnkle"][frame],
#             scale,
#         )
#         step_lengths.append(step_length)

#     # 重心速度
#     cogVelocities = calc_cog_velocity(cogs, cali_marker["origin"], 60, scale)

#     # print("cogs", cogs)
#     # print("velocities", velocities)
#     # print("trunk_angles", trunk_angles)
#     # print("hip_flexion_angles", hip_flexion_angles)
#     # print("step_lengths", step_lengths)
