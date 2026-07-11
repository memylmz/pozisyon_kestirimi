import cv2
import numpy as np


# Bize gelen kamera verilerine göre uygun kamera parametrelerini seçeceğiz.
# NOT (MATLAB -> OpenCV): MATLAB Camera Calibrator piksel indekslemesi 1-tabanlı
# (ilk pikselin merkezi (1,1)), OpenCV 0-tabanlı ((0,0)). Bu yüzden prensip
# noktası OpenCV'ye geçerken cx-1, cy-1 olur (odak uzaklığı ve distortion aynen
# kalır). Aşağıdaki cx,cy değerleri MATLAB PrincipalPoint'ten 1 çıkarılmış halidir.
# dist sırası OpenCV formatında: [k1, k2, p1, p2, k3] (MATLAB RadialDistortion=[k1 k2],
# TangentialDistortion=[p1 p2]).
CALIBRATIONS = [
    {
        "name": "RGB 1080p",
        "h": 1080,
        "w": 1920,
        "K": np.array([
            [1389.7, 0.0, 953.007],     # cx = 954.007 - 1
            [0.0, 1387.1, 557.896],     # cy = 558.896 - 1
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        "dist": np.array([0.1378, -0.2564, 0.0, 0.0, 0.0], dtype=np.float32)
    },
    {
        "name": "RGB 4K",
        "h": 3000,
        "w": 4000,
        "K": np.array([
            [2792.2, 0.0, 1987.0],      # cx = 1988.0 - 1
            [0.0, 2795.2, 1561.2],      # cy = 1562.2 - 1
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        "dist": np.array([0.0798, -0.1867, 0.0, 0.0, 0.0], dtype=np.float32)
    },
    {
        "name": "Thermal 640x512",
        "h": 512,
        "w": 640,
        "K": np.array([
            [731.7965, 0.0, 318.2367],  # cx = 319.2367 - 1
            [0.0, 732.0172, 250.2424],  # cy = 251.2424 - 1
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        "dist": np.array([-0.3507, 0.1137, 0.0, 0.0, 0.0], dtype=np.float32)
    }
]
def select_camera_calibration(frame_w, frame_h):
    for calib in CALIBRATIONS:
        if frame_w == calib["w"] and frame_h == calib["h"]:
            return calib, 1.0, 1.0, True

    frame_ratio = frame_w / frame_h

    best = min(
        CALIBRATIONS,
        key=lambda calib: abs(frame_ratio - (calib["w"] / calib["h"]))
    )

    sx = frame_w / best["w"]
    sy = frame_h / best["h"]
    same_aspect = abs(sx - sy) < 1e-3

    return best, sx, sy, same_aspect



def detect_features(
    gray,
    max_corners=1000,
    quality_level=0.01,
    min_distance=10,
    block_size=7,
    cell_size=120,
    max_corners_per_cell=12
):
    """
    Görüntüyü grid hücrelerine bölerek takip edilebilir feature/köşe noktalarını bulur.
    Böylece noktalar tek bir dokulu bölgede toplanmak yerine görüntüye daha dengeli yayılır.
    """
    h, w = gray.shape
    cell_features = []

    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = gray[y:y + cell_size, x:x + cell_size]

            pts = cv2.goodFeaturesToTrack(
                cell,
                maxCorners=max_corners_per_cell,
                qualityLevel=quality_level,
                minDistance=min_distance,
                blockSize=block_size
            )

            if pts is None:
                continue

            pts[:, 0, 0] += x
            pts[:, 0, 1] += y
            cell_features.append(pts)

    if not cell_features:
        return None

    selected = []
    feature_rank = 0

    while len(selected) < max_corners:
        added_feature = False

        for pts in cell_features:
            if feature_rank >= len(pts):
                continue

            selected.append(pts[feature_rank])
            added_feature = True

            if len(selected) >= max_corners:
                break

        if not added_feature:
            break

        feature_rank += 1

    return np.array(selected, dtype=np.float32).reshape(-1, 1, 2)
