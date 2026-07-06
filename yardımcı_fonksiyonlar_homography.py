import cv2
import numpy as np


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


def affine_depth_from_points(old_pts, new_pts, ransac_threshold=3.0):
    """
    Görüntüdeki genel büyüme/küçülmeyi similarity ölçeğiyle ölçer.
    log(scale) > 0 görüntünün büyüdüğünü, < 0 küçüldüğünü gösterir.
    """
    if len(old_pts) < 3 or len(new_pts) < 3:
        return np.nan, np.nan, 0

    affine_matrix, affine_mask = cv2.estimateAffinePartial2D(
        old_pts,
        new_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold
    )

    if affine_matrix is None:
        return np.nan, np.nan, 0

    a = affine_matrix[0, 0]
    b = affine_matrix[0, 1]
    image_scale = float(np.sqrt(a * a + b * b))

    if image_scale <= 0:
        return np.nan, np.nan, 0

    inliers = int(np.count_nonzero(affine_mask)) if affine_mask is not None else len(old_pts)
    return float(np.log(image_scale)), image_scale, inliers


def radial_depth_from_points(old_pts, new_pts, w, h, min_radius=30.0):
    """
    Takip edilen noktalar merkezden dışarı açılıyor mu, merkeze mi kapanıyor ölçer.
    Pozitif değer görüntü genişlemesini, negatif değer görüntü daralmasını gösterir.
    """
    if len(old_pts) < 3 or len(new_pts) < 3:
        return np.nan, 0

    old_2d = old_pts.reshape(-1, 2).astype(np.float64)
    new_2d = new_pts.reshape(-1, 2).astype(np.float64)

    center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    radial_vectors = old_2d - center
    radius_sq = np.sum(radial_vectors * radial_vectors, axis=1)
    valid = radius_sq > (min_radius * min_radius)

    if np.count_nonzero(valid) < 3:
        return np.nan, int(np.count_nonzero(valid))

    flows = new_2d - old_2d
    translation = np.median(flows[valid], axis=0)
    residual_flows = flows - translation

    radial_scale_change = (
        np.sum(residual_flows[valid] * radial_vectors[valid], axis=1) /
        radius_sq[valid]
    )

    return float(np.median(radial_scale_change)), int(np.count_nonzero(valid))


def essential_z_from_points(old_pts, new_pts, K, min_points=8, ransac_threshold=1.0):
    """
    Essential matrix üzerinden göreli translation yönünün z bileşenini üretir.
    Ölçek içermez; sadece frame'ler arası ileri/geri yön sinyali olarak düşünülmelidir.
    """
    if len(old_pts) < min_points or len(new_pts) < min_points:
        return np.nan, 0

    E, inlier_mask = cv2.findEssentialMat(
        old_pts,
        new_pts,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold
    )

    if E is None or inlier_mask is None:
        return np.nan, 0

    if E.shape != (3, 3):
        E = E[:3, :3]

    _, _R, t, pose_mask = cv2.recoverPose(E, old_pts, new_pts, K, mask=inlier_mask)
    inliers = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0

    if inliers < min_points:
        return np.nan, inliers

    return float(t[2, 0]), inliers


def essential_yaw_from_points(old_pts, new_pts, K, min_points=15, ransac_threshold=1.0):
    """
    Essential matrix + recoverPose ile kareler arası kameranın optik eksen (yaw)
    dönmesini derece cinsinden verir. Homography'nin aksine düzlemsel olmayan
    sahnede rotasyonu translation'dan (parallax) doğru ayırır; bu yüzden 3B yapı
    üzerinde heading kaynağı olarak homography açısından daha güvenilirdir.

    Uyarı: Sahne neredeyse tam düzlemsel + hareket küçükse essential matrix
    dejenere olur; bu durumda çağıran tarafın homography açısına düşmesi beklenir.
    """
    if len(old_pts) < min_points or len(new_pts) < min_points:
        return np.nan, 0

    E, inlier_mask = cv2.findEssentialMat(
        old_pts,
        new_pts,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold
    )

    if E is None or inlier_mask is None:
        return np.nan, 0

    if E.shape != (3, 3):
        E = E[:3, :3]

    _, R, _t, pose_mask = cv2.recoverPose(E, old_pts, new_pts, K, mask=inlier_mask)
    inliers = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0

    if inliers < min_points:
        return np.nan, inliers

    # Optik eksen (kamera Z) etrafındaki dönme = yaw. Rodrigues vektörünün Z bileşeni.
    rvec, _ = cv2.Rodrigues(R)
    yaw_deg = float(np.degrees(rvec[2, 0]))

    if not np.isfinite(yaw_deg):
        return np.nan, inliers

    return yaw_deg, inliers


def homography_z_from_decomposition(H, K, previous_z=None):
    """
    Homography decomposition ile t/d vektörlerinin z bileşeninden sinyal üretir.
    Birden fazla çözüm geldiği için önceki frame'e en yakın z adayı seçilir.
    """
    retval, _Rs, ts, _normals = cv2.decomposeHomographyMat(H, K)

    if retval <= 0 or ts is None:
        return np.nan, 0

    z_candidates = np.array([float(t.reshape(3)[2]) for t in ts], dtype=np.float64)

    if z_candidates.size == 0:
        return np.nan, 0

    if previous_z is not None and np.isfinite(previous_z):
        best_index = int(np.argmin(np.abs(z_candidates - previous_z)))
    else:
        best_index = int(np.argmax(np.abs(z_candidates)))

    return float(z_candidates[best_index]), int(z_candidates.size)


def vote_depth_signals(signals, thresholds):
    """
    Sinyallerin işaretlerine göre 2/3 benzeri basit oylama yapar.
    Pozitif/negatif yönün gerçek dünyadaki anlamı video üzerinde kalibre edilmelidir.
    """
    positive_votes = 0
    negative_votes = 0

    for name, value in signals.items():
        threshold = thresholds.get(name, 0.0)

        if value is None or not np.isfinite(value):
            continue

        if value > threshold:
            positive_votes += 1
        elif value < -threshold:
            negative_votes += 1

    if positive_votes >= 2:
        vote = "positive"
    elif negative_votes >= 2:
        vote = "negative"
    else:
        vote = "uncertain"

    return vote, positive_votes, negative_votes



# Bize gelen kamera verilerine göre uygun kamera parametrelerini seçeceğiz.
CALIBRATIONS = [
    {
        "name": "RGB 1080p",
        "h": 1080,
        "w": 1920,
        "K": np.array([
            [1389.7, 0.0, 954.007],
            [0.0, 1387.1, 558.896],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        "dist": np.array([0.1378, -0.2564, 0.0, 0.0, 0.0], dtype=np.float32)
    },
    {
        "name": "RGB 4K",
        "h": 3000,
        "w": 4000,
        "K": np.array([
            [2792.2, 0.0, 1988.0],
            [0.0, 2795.2, 1562.2],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        "dist": np.array([0.0798, -0.1867, 0.0, 0.0, 0.0], dtype=np.float32)
    },
    {
        "name": "Thermal 640x512",
        "h": 512,
        "w": 640,
        "K": np.array([
            [731.7965, 0.0, 319.2367],
            [0.0, 732.0172, 251.2424],
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
