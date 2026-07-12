import cv2
import numpy as np

def write_homography_log(
    writer,
    frame_index,
    status,
    #dx_px=0.0,
    #dy_px=0.0,
    #dx_h=0.0,
    #dy_h=0.0,
    #angle_deg=0.0,
    #scale=1.0,
    global_dx_px=0.0,
    global_dy_px=0.0,
    global_angle_deg=0.0,
    global_scale=1.0,
    **_unused_metrics
    #fb_count=0,
    #homography_inliers=0,
    #clean_inliers=0
):
    writer.writerow([
        frame_index,
        status,
        #dx_px,
        #dy_px,
        #dx_h,
        #dy_h,
        #angle_deg,
        #scale,
        global_dx_px,
        global_dy_px,
        global_angle_deg,
        global_scale,
        #fb_count,
        #homography_inliers,
        #clean_inliers
    ])


def angle_scale_from_homography(H, w, h, line_len=100):
    """
    Homography'nin görüntü merkezindeki yatay bir çizgiyi
    ne kadar döndürdüğünü ve ölçeklediğini hesaplar.
    """

    ref_points = np.array([
        [[w / 2, h / 2]],
        [[w / 2 + line_len, h / 2]]
    ], dtype=np.float32)

    warped_ref = cv2.perspectiveTransform(ref_points, H)

    x1, y1 = warped_ref[0, 0]
    x2, y2 = warped_ref[1, 0]

    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)

    scale = np.hypot(x2 - x1, y2 - y1) / line_len

    return angle_deg, scale



# Bize gelen kamera verilerine göre uygun kamera parametrelerini seçeceğiz.
# NOT (MATLAB -> OpenCV): MATLAB Camera Calibrator 1-tabanlı piksel indeksler
# (ilk piksel merkezi (1,1)), OpenCV 0-tabanlı. Prensip noktası OpenCV'ye
# geçerken cx-1, cy-1 olur (odak ve distortion aynen kalır). Aşağıdaki cx,cy
# MATLAB PrincipalPoint'ten 1 çıkarılmış halidir.
# Eski kamera parametreleri (silinmedi; gerektiğinde geri alınabilir):


"""CALIBRATIONS = [
     {
         "name": "RGB 1080p",
         "h": 1080,
         "w": 1920,
         "K": np.array([
             [1389.7, 0.0, 953.007],
             [0.0, 1387.1, 557.896],
             [0.0, 0.0, 1.0]

         ], dtype=np.float32),
         "dist": np.array([0.1378, -0.2564, 0.0, 0.0, 0.0], dtype=np.float32)
     },
     {
         "name": "RGB 4K",
         "h": 3000,
         "w": 4000,
         "K": np.array([

             [2792.2, 0.0, 1987.0],
             [0.0, 2795.2, 1561.2],
             [0.0, 0.0, 1.0]
         ], dtype=np.float32),
         "dist": np.array([0.0798, -0.1867, 0.0, 0.0, 0.0], dtype=np.float32)
     },
     {
         "name": "Thermal 640x512",
         "h": 512,
         "w": 640,
         "K": np.array([
             [731.7965, 0.0, 318.2367],
             [0.0, 732.0172, 250.2424],
             [0.0, 0.0, 1.0]
            ], dtype=np.float32),
         "dist": np.array([-0.3507, 0.1137, 0.0, 0.0, 0.0], dtype=np.float32)
     }
 ]"""

# Yeni MATLAB Camera Calibrator parametreleri.
# TangentialDistortion [0, 0] ve üçüncü radyal katsayı verilmediği için
# OpenCV sıralaması [k1, k2, p1, p2, k3] içinde kalan değerler sıfırdır.
CALIBRATIONS = [
    {
        "name": "RGB 4000x3000",
       "h": 3000,
        "w": 4000,
        "K": np.array([
            [2792.2, 0.0, 1987.0],      # MATLAB cx = 1988.0
            [0.0, 2795.2, 1561.2],      # MATLAB cy = 1562.2
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        "dist": np.array([0.0798, -0.1867, 0.0, 0.0, 0.0], dtype=np.float32)
    },
    {
        "name": "Thermal 640x512",
        "h": 512,
        "w": 640,
        "K": np.array([
            [731.7965, 0.0, 318.2367],  # MATLAB cx = 319.2367
            [0.0, 732.0172, 250.2424],  # MATLAB cy = 251.2424
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


def filter_by_homography_error(H, old_pts, new_pts, max_error=2.5):
    """
    Homography ile eski noktaları yeni frame'e projekte eder.
    Projeksiyon hatası büyük olan noktaları eler.
    """
    old_pts_reshaped = old_pts.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(old_pts_reshaped, H).reshape(-1, 2)

    new_pts_2d = new_pts.reshape(-1, 2)

    errors = np.linalg.norm(projected - new_pts_2d, axis=1)

    valid = errors < max_error

    return old_pts[valid], new_pts[valid], errors[valid]


def create_sample_points(w, h):
    """
    Tek merkez noktası yerine görüntü üzerinde 9 nokta kullanıyoruz.
    Böylece dönme/perspektif etkisine karşı daha dengeli hareket ölçülür.
    """
    return np.array([
        [[w * 0.25, h * 0.25]],
        [[w * 0.50, h * 0.25]],
        [[w * 0.75, h * 0.25]],

        [[w * 0.25, h * 0.50]],
        [[w * 0.50, h * 0.50]],
        [[w * 0.75, h * 0.50]],

        [[w * 0.25, h * 0.75]],
        [[w * 0.50, h * 0.75]],
        [[w * 0.75, h * 0.75]],
    ], dtype=np.float32)
