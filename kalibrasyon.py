"""
Kalibrasyon yardımcıları.

Fikir: ilk N frame boyunca GT (CSV) konumunu DOĞRU kabul ederiz. Aynı anda kendi
VO tahminimizi (iç birim: irtifa-normalize piksel) biriktiririz. N'inci frame'de
iç-birim yörüngeyi metrik GT yörüngesine hizalayan tek bir 2B dönüşüm buluruz:

    metre = A @ ic_birim + b        (A: 2x2 ölçek+dönme+yansıma, b: öteleme)

Bu dönüşüm hem ÖLÇEĞİ (metre/piksel) hem de kamera-0 çerçevesi ile GT dünya
çerçevesi arasındaki DÖNME/YANSIMA farkını çözer. Sonra 450'den sonra kendi
tahminimize bu dönüşümü uygulayıp metrik konum üretiriz.
"""

import csv
import numpy as np
import cv2


def load_gt_translations(csv_path):
    """CSV'den frame sırasına göre (x, y, z) metre dizilerini okur.

    Beklenen kolonlar: translation_x, translation_y, translation_z, frame_numbers
    """
    xs, ys, zs = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["translation_x"]))
            ys.append(float(row["translation_y"]))
            zs.append(float(row["translation_z"]))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64), np.array(zs, dtype=np.float64)


def fit_calibration(est_xy, gt_xy, ransac_thresh_m=3.0):
    """İç-birim tahmin yörüngesini metrik GT'ye hizalayan 2x3 dönüşümü bulur.

    est_xy, gt_xy: (N, 2) diziler. Dönüş: (M 2x3, inliers).
    estimateAffine2D tam afin olduğu için ölçek + dönme + YANSIMA (görüntü y
    ekseni aşağı olduğundan gerekebilir) + öteleme hepsini yakalar. RANSAC ile
    kötü VO frame'lerini eler.
    """
    est = np.asarray(est_xy, dtype=np.float32).reshape(-1, 1, 2)
    gt = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffine2D(
        est, gt, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_m
    )
    return M, inliers


def calibration_report(M, est_xy, gt_xy):
    """Kalibrasyonun kalitesini raporlar: ölçek, dönme, RMSE (metre)."""
    est = np.asarray(est_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)
    A = M[:, :2]
    scale = float(np.sqrt(abs(np.linalg.det(A))))          # metre / iç-birim
    rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    reflected = bool(np.linalg.det(A) < 0)
    proj = est @ A.T + M[:, 2]
    err = np.linalg.norm(proj - gt, axis=1)
    return {
        "scale_m_per_unit": scale,
        "rotation_deg": rot_deg,
        "reflected": reflected,
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "mean_err_m": float(np.mean(err)),
        "max_err_m": float(np.max(err)),
        "n_points": int(len(est)),
    }


def apply_calibration(M, x, y):
    """İç-birim (x, y) -> metrik (x, y)."""
    p = M[:, :2] @ np.array([x, y], dtype=np.float64) + M[:, 2]
    return float(p[0]), float(p[1])
