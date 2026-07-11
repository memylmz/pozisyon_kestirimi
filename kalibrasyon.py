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


def fit_z_calibration(est_z_raw, gt_z, min_span=0.03):
    """Ham ölçek tabanlı z sinyalini GT z değerine hizalar.

    Model:
        z_metre = a * z_raw + b

    Burada z_raw genelde 1 / cumulative_scale olur. Böylece GT z'nin başlangıç
    ofseti ve işareti ne olursa olsun tek boyutlu doğrusal hizalama yapılır.

    KOŞULLANMA: kalibrasyon penceresinde irtifa gerçekten değişmeliyse z_raw
    yayılımı yeterli olmalı. İrtifa neredeyse sabitken (bu penceredeki z_raw
    yayılımı çoğunlukla VO ölçek drift'inden gelir) slope 'a' gürültüden fit
    edilir ve tüm uçuşun ~100x daha geniş z_raw aralığına EKSTRAPOLE edilince z
    aşırı ölçeklenir. Bu yüzden yayılım min_span'dan (fit_altitude_from_vertical
    ile aynı 0.03 eşiği) küçükse None döneriz; çağıran sağlam fallback'e
    (H0 * (z_raw - 1)) düşer.
    """
    est = np.asarray(est_z_raw, dtype=np.float64)
    gt = np.asarray(gt_z, dtype=np.float64)
    valid = np.isfinite(est) & np.isfinite(gt)
    est = est[valid]
    gt = gt[valid]

    if len(est) < 2 or np.ptp(est) < min_span:
        return None

    a, b = np.polyfit(est, gt, 1)
    return float(a), float(b)


def apply_z_calibration(params, z_raw):
    """Ham z sinyalini metrik z değerine çevirir."""
    a, b = params
    return float(a * z_raw + b)


def z_calibration_report(params, est_z_raw, gt_z):
    """Z kalibrasyonu için hata istatistikleri üretir."""
    est = np.asarray(est_z_raw, dtype=np.float64)
    gt = np.asarray(gt_z, dtype=np.float64)
    pred = params[0] * est + params[1]
    err = pred - gt
    return {
        "scale": float(params[0]),
        "offset": float(params[1]),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "mean_abs_err_m": float(np.mean(np.abs(err))),
        "max_abs_err_m": float(np.max(np.abs(err))),
        "n_points": int(len(est)),
    }


def fit_altitude_from_vertical(cum_scales, gt_z):
    """Başlangıç irtifası H0'ı DİKEY kanaldan çıkarır.

    cumulative_scale S_k = H0 / H_k olduğundan, GT'nin dikey yer değiştirmesi:
        GT_z[k] = H_k - H0 = H0 * (1/S_k - 1)
    Bu H0'da lineer; u = 1/S - 1 dersek en-küçük-kareler (orijinden):
        H0 = (u . z) / (u . u)
    Yatay 2B fit dejenere olsa bile bu bağımsız çalışır. Koşul: pencerede
    irtifa gerçekten değişmeli (u yayılımı yeterli olmalı).

    Dönüş: (H0, u_span). u_span küçükse (örn. <0.03) sonuç güvenilmez.
    """
    S = np.asarray(cum_scales, dtype=np.float64)
    z = np.asarray(gt_z, dtype=np.float64)
    u = 1.0 / S - 1.0
    denom = float(u @ u)
    if denom < 1e-12:
        return None, 0.0
    H0 = float((u @ z) / denom)
    u_span = float(u.max() - u.min())
    return H0, u_span
