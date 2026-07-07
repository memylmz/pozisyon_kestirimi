import numpy as np
import cv2
import time
import csv
import re
from pathlib import Path
from yardımcı_fonksiyonlar_homography import *


BASE_DIR = Path(__file__).resolve().parent
ESTIMATED_MOTION_CSV = BASE_DIR / "estimated_motion.csv"
REAL_TRANSLATION_CSV = Path(
    "/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1_translation.csv"
)
CALIBRATION_FRAME_LIMIT = 450
# Z (irtifa) ölçeği yatay hareketten çok daha yavaş gözlemlenir: drone ilk ~1700 kare
# neredeyse düz uçtuğu için 450'lik pencerede irtifa aralığı ~0.34m (çoğu gürültü) kalıyor.
# Bu yüzden z için ayrı ve daha geniş bir kalibrasyon penceresi kullanıyoruz.
Z_CALIBRATION_FRAME_LIMIT = 2000
# Bu pencerede en az bu kadar (m) gerçek irtifa açıklığı görülmeden z-ölçeğine güvenilmez;
# altındaysa z "güvenilmez" işaretlenir ve sessizce ekstrapole edilmez.
Z_MIN_CALIB_SPAN_M = 3.0
# xy affine ölçeği de 450 karelik pencerede kalibre edilince sonrasında ölçek drifti
# yüzünden yol fazla uzun tahmin ediliyor (loop'lar geniş). Daha uzun pencere ölçeği
# ortalayıp drifti azaltır (offline: toplam uzunluk oranı 0.80 -> 0.95). Rebase noktası
# (anchor) yine CALIBRATION_FRAME_LIMIT'te; sadece affine fit penceresi genişliyor.
XY_CALIBRATION_FRAME_LIMIT = 2000
# Yaw bias'ı (deg/kare) da yavaş gözlemlenir: ~0.02°/kare'lik sabit bir bias 450 karelik
# pencerede fark edilemez (residual iyileşmesi ~%1) ama 8500 karede 100°+ birikir. Bu yüzden
# yaw-bias kestirimini de daha uzun bir pencerede yapıyoruz (2000 karede iyileşme ~%24).
YAW_CALIBRATION_FRAME_LIMIT = 2000
MIN_CALIBRATION_ROWS = 20
USE_INCREMENTAL_REAL_CSV = False
Z_MIN_CLEAN_INLIERS = 60
Z_MAX_ROTATION_DEG = 1.5
Z_MAX_DELTA_PER_FRAME = 0.003
Z_HISTORY_SIZE = 30
Z_OUTLIER_MAD_FACTOR = 5.0


def read_real_motion(csv_path):
    if not csv_path.exists():
        print("Gerçek hareket CSV dosyası bulunamadı; hareket ölçeksiz gösterilecek.")
        return None

    required_columns = [
        "translation_x",
        "translation_y",
        "translation_z",
        "frame_numbers",
    ]
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as real_file:
        reader = csv.DictReader(real_file)
        missing_columns = [
            col for col in required_columns
            if col not in (reader.fieldnames or [])
        ]

        if missing_columns:
            print("Gerçek hareket CSV dosyasında eksik kolonlar var:", missing_columns)
            return None

        for row in reader:
            frame_match = re.search(r"(\d+)", str(row["frame_numbers"]))

            if frame_match is None:
                continue

            try:
                parsed_row = {
                    "frame": int(frame_match.group(1)),
                    "translation_x": float(row["translation_x"]),
                    "translation_y": float(row["translation_y"]),
                    "translation_z": float(row["translation_z"]),
                }
            except (TypeError, ValueError):
                continue

            rows.append(parsed_row)

    rows.sort(key=lambda item: item["frame"])
    real_motion = {}
    cumulative_position = np.zeros(3, dtype=np.float64)

    for row in rows:
        translation = np.array([
            row["translation_x"],
            row["translation_y"],
            row["translation_z"],
        ], dtype=np.float64)

        if USE_INCREMENTAL_REAL_CSV:
            cumulative_position += translation
            real_motion[row["frame"]] = cumulative_position.copy()
        else:
            real_motion[row["frame"]] = translation

    print("Gerçek hareket CSV yüklendi:", csv_path)
    print("Gerçek hareket frame sayısı:", len(real_motion))
    return real_motion


def fit_motion_calibration(estimated_rows, real_motion, anchor_frame, xy_anchor_frame, z_anchor_frame):
    # xy affine ölçeği tek bir kısa pencerede (450) kalibre edilince, pencere dışında
    # monoküler ölçek drifti yüzünden yol giderek fazla uzun oluyor (loop'lar şişiyor).
    # Daha geniş pencere (xy_anchor_frame) ölçeği ortalayıp bu drifti belirgin azaltır.
    matched_rows = [
        row for row in estimated_rows
        if (
            row["status"] == "ok" and
            row["frame"] <= xy_anchor_frame and
            row["frame"] in real_motion
        )
    ]

    if len(matched_rows) < MIN_CALIBRATION_ROWS:
        return None

    estimated_xy = np.array(
        [[row["map_x_px"], row["map_y_px"]] for row in matched_rows],
        dtype=np.float64
    )
    real_xyz = np.array(
        [real_motion[row["frame"]] for row in matched_rows],
        dtype=np.float64
    )

    xy_design = np.column_stack([
        estimated_xy[:, 0],
        estimated_xy[:, 1],
        np.ones(len(estimated_xy)),
    ])

    params_x, _, _, _ = np.linalg.lstsq(xy_design, real_xyz[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(xy_design, real_xyz[:, 1], rcond=None)

    # Z için AYRI ve daha geniş pencere: irtifa değişimi yatay hareketten çok geç başlar.
    z_matched_rows = [
        row for row in estimated_rows
        if (
            row["frame"] <= z_anchor_frame and
            row["frame"] in real_motion and
            row.get("z_status") in ("ok", "deadzone")
        )
    ]

    if len(z_matched_rows) < MIN_CALIBRATION_ROWS:
        z_matched_rows = [
            row for row in estimated_rows
            if (
                row["status"] == "ok" and
                row["frame"] <= z_anchor_frame and
                row["frame"] in real_motion
            )
        ]

    # Z sadece birikmiş ölçekten (irtifa ile lineer feature) tahmin edilir;
    # yatay konum (map_x/map_y) girdi olarak dahil edilmez, aksi halde overfit olur.
    estimated_z_features = np.array(
        [
            [row["z_unscaled"], 1.0]
            for row in z_matched_rows
        ],
        dtype=np.float64
    )
    real_z = np.array(
        [real_motion[row["frame"]][2] for row in z_matched_rows],
        dtype=np.float64
    )
    params_z, _, _, _ = np.linalg.lstsq(estimated_z_features, real_z, rcond=None)

    # Gözlemlenebilirlik guardı: pencerede yeterli gerçek irtifa açıklığı yoksa
    # z-ölçeği gürültüden çıkarılmış demektir; güvenilmez işaretle, ekstrapole etme.
    z_span = float(real_z.max() - real_z.min()) if len(real_z) else 0.0
    z_reliable = (
        len(z_matched_rows) >= MIN_CALIBRATION_ROWS and
        z_span >= Z_MIN_CALIB_SPAN_M
    )

    anchor_candidates = [
        row for row in estimated_rows
        if row["frame"] <= anchor_frame and row["frame"] in real_motion
    ]

    if not anchor_candidates:
        return None

    anchor_row = anchor_candidates[-1]
    anchor_input = np.array(
        [anchor_row["map_x_px"], anchor_row["map_y_px"], 1.0],
        dtype=np.float64
    )
    anchor_pred = np.array([
        float(anchor_input @ params_x),
        float(anchor_input @ params_y),
        float(np.array([
            anchor_row["z_unscaled"],
            1.0,
        ], dtype=np.float64) @ params_z),
    ], dtype=np.float64)

    pred_x = xy_design @ params_x
    pred_y = xy_design @ params_y
    pred_z = estimated_z_features @ params_z
    rmse = np.array([
        np.sqrt(np.mean((pred_x - real_xyz[:, 0]) ** 2)),
        np.sqrt(np.mean((pred_y - real_xyz[:, 1]) ** 2)),
        np.sqrt(np.mean((pred_z - real_z) ** 2)),
    ], dtype=np.float64)

    return {
        "params_x": params_x,
        "params_y": params_y,
        "params_z": params_z,
        "anchor_frame": anchor_row["frame"],
        "anchor_real": real_motion[anchor_row["frame"]],
        "anchor_pred": anchor_pred,
        "row_count": len(matched_rows),
        "z_row_count": len(z_matched_rows),
        "z_span": z_span,
        "z_reliable": z_reliable,
        "rmse": rmse,
    }


def estimate_real_position(map_x, map_y, z_unscaled, calibration):
    if calibration is None:
        return None

    xy_input = np.array([map_x, map_y, 1.0], dtype=np.float64)
    pred_x = float(xy_input @ calibration["params_x"])
    pred_y = float(xy_input @ calibration["params_y"])

    if calibration.get("z_reliable", False):
        z_input = np.array([z_unscaled, 1.0], dtype=np.float64)
        pred_z = float(z_input @ calibration["params_z"])
    else:
        # Z güvenilmez: ekstrapole etme, irtifayı anchor değerinde tut (pred - anchor_pred = 0).
        pred_z = float(calibration["anchor_pred"][2])

    predicted = np.array([pred_x, pred_y, pred_z], dtype=np.float64)
    return calibration["anchor_real"] + (predicted - calibration["anchor_pred"])


def select_trajectory_point(frame_no, real_position, map_x, map_y):
    if (
        real_motion_by_frame is not None and
        frame_no <= CALIBRATION_FRAME_LIMIT and
        frame_no in real_motion_by_frame
    ):
        real_csv_position = real_motion_by_frame[frame_no]
        return (
            float(real_csv_position[0]),
            float(real_csv_position[1]),
            "csv"
        )

    if real_position is not None:
        return (
            float(real_position[0]),
            float(real_position[1]),
            "scaled"
        )

    return float(map_x), float(map_y), "px"


def reconstruct_biased_positions(samples, yaw_bias_deg):
    """
    Kaydedilmiş kare-başı (zaten heading ile döndürülmüş) yer değiştirmeleri,
    sabit bir per-frame yaw bias'ı çıkarılmış gibi yeniden entegre eder.
    Bias, o kareye kadar yapılmış heading güncelleme sayısıyla orantılı bir
    ek dönme (R(-bias*count)) olarak uygulanır.
    """
    beta = np.deg2rad(yaw_bias_deg)
    cx = 0.0
    cy = 0.0
    reconstructed = {}

    for frame_no, rotated_dx, rotated_dy, update_count in samples:
        angle = -beta * update_count
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        cx += cos_a * rotated_dx - sin_a * rotated_dy
        cy += sin_a * rotated_dx + cos_a * rotated_dy
        reconstructed[frame_no] = (cx, cy)

    return reconstructed


def yaw_bias_affine_residual(reconstructed, real_motion):
    """
    Yeniden kurulan xy yolunu gerçek CSV yoluna en iyi oturtan 2D affine
    (döndürme+ölçek+kayma) sonrası kalan RMSE'yi döndürür. Bias yanlışsa yol
    global affine ile hizalanamaz ve residual büyür.
    """
    frames = [f for f in reconstructed if f in real_motion]

    if len(frames) < MIN_CALIBRATION_ROWS:
        return np.inf

    design = np.array(
        [[reconstructed[f][0], reconstructed[f][1], 1.0] for f in frames],
        dtype=np.float64
    )
    real_x = np.array([real_motion[f][0] for f in frames], dtype=np.float64)
    real_y = np.array([real_motion[f][1] for f in frames], dtype=np.float64)

    params_x, _, _, _ = np.linalg.lstsq(design, real_x, rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(design, real_y, rcond=None)

    err_x = design @ params_x - real_x
    err_y = design @ params_y - real_y
    return float(np.sqrt(np.mean(err_x * err_x + err_y * err_y)))


def estimate_yaw_bias(samples, real_motion):
    """
    Kalibrasyon penceresindeki artışlar üzerinde sabit per-frame yaw bias'ını
    (deg/kare) 1B arama ile bulur: affine-sonrası residual'ı minimize eder.
    (best_bias, best_residual, base_residual) döndürür.
    """
    base_residual = yaw_bias_affine_residual(
        reconstruct_biased_positions(samples, 0.0), real_motion
    )

    if len(samples) < MIN_CALIBRATION_ROWS or not np.isfinite(base_residual):
        return 0.0, base_residual, base_residual

    best_bias = 0.0
    best_residual = base_residual

    # Kaba tarama, sonra en iyi nokta çevresinde ince tarama.
    for grid in (np.linspace(-0.1, 0.1, 41), None):
        if grid is None:
            grid = np.linspace(best_bias - 0.005, best_bias + 0.005, 41)
        for candidate in grid:
            residual = yaw_bias_affine_residual(
                reconstruct_biased_positions(samples, float(candidate)),
                real_motion
            )
            if residual < best_residual:
                best_residual = residual
                best_bias = float(candidate)

    return best_bias, best_residual, base_residual


real_motion_by_frame = read_real_motion(REAL_TRANSLATION_CSV)
motion_calibration = None
calibration_done = False
calibration_text = "Calib: waiting"






# Video yolu
cap = cv2.VideoCapture("/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1.MP4")
#cap = cv2.VideoCapture(
#    "/Users/mehmetyilmaz/Desktop/2025_HYZ_Ornek_Veriler/Ornek_Veri_Gunduz_Kamera_VO.MP4")


lk_params = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


ret, old_frame = cap.read()
if not ret:
    raise RuntimeError("Video okunamadı.")


h, w = old_frame.shape[:2]
print("Frame shape:", old_frame.shape)


selected_calib, sx, sy, same_aspect = select_camera_calibration(w, h)


K = selected_calib["K"].copy()
K[0, 0] *= sx   # fx
K[0, 2] *= sx   # cx
K[1, 1] *= sy   # fy
K[1, 2] *= sy   # cy

dist_coeffs = selected_calib["dist"]


print("Selected calibration:", selected_calib["name"])
print("scale_x:", sx)
print("scale_y:", sy)
print("Scaled K:\n", K)


if not same_aspect:
    print("UYARI: Video çözünürlüğü kayıtlı kalibrasyonlarla birebir veya aynı oranda eşleşmiyor.")
    print("UYARI: En yakın kalibrasyon seçildi; crop/letterbox varsa K, undistortion ve hareket hesabı sapabilir.")


# Undistortion haritasını bir kez hesapla
map1, map2 = cv2.initUndistortRectifyMap(
    K,
    dist_coeffs,
    None,
    K,
    (w, h),
    cv2.CV_16SC2
)


old_frame = cv2.remap(old_frame, map1, map2, cv2.INTER_LINEAR)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

h, w = old_gray.shape

sample_points = create_sample_points(w, h)

p0 = detect_features(old_gray)

if p0 is None:
    raise RuntimeError("İlk frame üzerinde takip edilecek feature bulunamadı.")


# Görselleştirme ayarları
arrow_color = (0, 0, 255)
point_color = (0, 255, 255)
arrow_thickness = 3
arrow_tip_length = 0.45
arrow_scale = 3.0
min_motion_threshold = 0.5


# Filtre / Homography ayarları
min_homography_points = 30
fb_error_threshold = 1.5
homography_ransac_threshold = 3.0
homography_reprojection_error = 2.5
affine_ransac_threshold = 3.0

z_deadzone = 0.0002
z_smoothing_alpha = 0.25
z_direction = -1.0


# Global Homography
global_H = np.eye(3, dtype=np.float64)

global_dx_px = 0.0
global_dy_px = 0.0

global_angle_deg = 0.0
global_scale = 1.0
z_position_unscaled = 0.0
filtered_z_delta_unscaled = 0.0
z_signal_history = []
z_update_status = "initial"

homography_text = "Homography: waiting"
z_text = "Z unscaled: 0.00000"

prev_time = time.time()

<<<<<<< HEAD
# Harita çözünürlüğü: yol tüm alanı doldurunca küçük hareketler alt-piksel kalıp
# görünmez oluyordu; büyüterek ince detay (küçük dönüş/hareket) görünür hale geliyor.
map_w = 600
map_h = 600
map_center_x = map_w // 2
map_center_y = map_h // 2
draw_scale = 1.0
if real_motion_by_frame is not None and 0 in real_motion_by_frame:
    initial_real_position = real_motion_by_frame[0]
    trajectory = [(float(initial_real_position[0]), float(initial_real_position[1]))]
    trajectory_source = "csv"
else:
    trajectory = [(0.0, 0.0)]
    trajectory_source = "px"
map_x_px = 0.0
map_y_px = 0.0
map_heading_deg = 0.0
# Yaw birikimi: hard deadband küçük tutarlı dönüşleri komple atıyordu (yavaş dönüş = eşik
# altı küçük açılar dizisi -> hepsi silinip dönüş kayboluyordu). Bunun yerine EMA ile
# gürültüyü bastırıp her karede biriktiriyoruz; sadece bariz tek-kare sıçramalarını eliyoruz.
filtered_angle_deg = 0.0
# EMA yumuşatma katsayısı: yüksek = daha az yumuşatma, daha keskin dönüşler (ama daha çok
# jitter). 0.3 keskin/kısa dönüşleri ~6 kareye yayıp yuvarlıyordu; bias+outlier düzeltmeleri
# artık jitter'ı tuttuğu için 0.6'ya çıkarıp dönüş tepkisini keskinleştiriyoruz.
yaw_smoothing_alpha = 0.6
yaw_outlier_deg = 10.0

# Yaw drift düzeltmesi: heading saf görsel integrasyon olduğu için küçük bir sistematik
# bias zamanla birikip yörüngeyi büküyor. CSV'de yaw ground-truth'u olmadığından, ilk
# CALIBRATION_FRAME_LIMIT karede kestirilen yolu gerçek CSV yoluna en iyi oturtan sabit
# per-frame yaw bias'ını (deg/kare) bulup sonrasında her kareden çıkarıyoruz.
motion_increment_samples = []   # (frame, rotated_dx, rotated_dy, heading_update_count)
heading_update_count = 0
yaw_bias_deg = 0.0
yaw_bias_done = False

windows_positioned = False

# İlk video karesi (old_frame) CSV'de frame_000000; estimated frame_index'i de 0'dan
# başlatarak gerçek CSV kare numaralarıyla birebir hizalıyoruz (önceki off-by-one düzeltmesi).
frame_index = 0
estimated_motion_file = open(ESTIMATED_MOTION_CSV, "w", newline="", encoding="utf-8")
estimated_motion_writer = csv.writer(estimated_motion_file)
estimated_motion_writer.writerow([
    "frame",
=======
log_file = open("homography_features.csv", "w", newline="", encoding="utf-8")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "frame_index",
>>>>>>> parent of cd5c7ec (dönme mantığında iyileştirmeler yapıldı ve trajectory haritası eklendi)
    "status",
    "map_x_px",
    "map_y_px",
    "z_unscaled",
    "z_status",
])
estimated_motion_rows = []


def record_estimated_motion(status):
    # z_position_unscaled birikmiş log-ölçek (~ -log(Z/Z0)) tutar.
    # Nadir kamerada irtifa ile LİNEER olan büyüklük exp(birikim) = Z0/Z olduğundan
    # kalibrasyona bu üstel feature'ı veriyoruz; böylece lstsq lineer modeli doğru olur.
    z_feature = float(np.exp(z_position_unscaled))
    estimated_motion_rows.append({
        "frame": frame_index,
        "status": status,
        "map_x_px": map_x_px,
        "map_y_px": map_y_px,
        "z_unscaled": z_feature,
        "z_status": z_update_status,
    })
    estimated_motion_writer.writerow([
        frame_index,
        status,
        map_x_px,
        map_y_px,
        z_feature,
        z_update_status,
    ])
    try_build_online_calibration()


def try_build_online_calibration():
    global motion_calibration
    global calibration_done
    global calibration_text

    # Z penceresi xy'den geniş olduğu için kalibrasyon z limitine kadar güncellenmeye devam eder.
    if calibration_done and frame_index > Z_CALIBRATION_FRAME_LIMIT:
        return

    if real_motion_by_frame is None:
        calibration_text = "Calib: no real csv"
        return

    ok_count = sum(
        1 for row in estimated_motion_rows
        if (
            row["status"] == "ok" and
            row["frame"] <= CALIBRATION_FRAME_LIMIT and
            row["frame"] in real_motion_by_frame
        )
    )

    if ok_count < MIN_CALIBRATION_ROWS:
        calibration_text = (
            f"Calib: learning wait ok:{ok_count}/{MIN_CALIBRATION_ROWS} "
            f"frame:{frame_index}"
        )
        return

    anchor_frame = min(frame_index, CALIBRATION_FRAME_LIMIT)
    xy_anchor_frame = min(frame_index, XY_CALIBRATION_FRAME_LIMIT)
    z_anchor_frame = min(frame_index, Z_CALIBRATION_FRAME_LIMIT)
    calibration = fit_motion_calibration(
        estimated_motion_rows,
        real_motion_by_frame,
        anchor_frame,
        xy_anchor_frame,
        z_anchor_frame
    )

    if calibration is None:
        calibration_text = f"Calib: not enough ok frames ({ok_count})"
        return

    motion_calibration = calibration
    calibration_done = frame_index >= Z_CALIBRATION_FRAME_LIMIT
    rmse = calibration["rmse"]
    mode = "final" if calibration_done else "live"
    z_flag = "OK" if calibration["z_reliable"] else "UNREL"
    calibration_text = (
        f"Calib {mode} xy:{calibration['row_count']} z:{calibration['z_row_count']} "
        f"rmse x:{rmse[0]:.2f} y:{rmse[1]:.2f} z:{rmse[2]:.2f} "
        f"zspan:{calibration['z_span']:.1f}m {z_flag}"
    )
    if calibration_done or frame_index % 30 == 0:
        print(calibration_text)


def robust_z_delta(raw_z_delta, h_status, clean_count, frame_angle_deg):
    if h_status != "ok":
        return 0.0, "hold:h"

    if clean_count < Z_MIN_CLEAN_INLIERS:
        return 0.0, "hold:pts"

    if abs(frame_angle_deg) > Z_MAX_ROTATION_DEG:
        return 0.0, "hold:rot"

    if not np.isfinite(raw_z_delta):
        return 0.0, "hold:nan"

    z_delta = z_direction * raw_z_delta

    if abs(z_delta) < z_deadzone:
        z_signal_history.append(0.0)
        if len(z_signal_history) > Z_HISTORY_SIZE:
            z_signal_history.pop(0)
        return 0.0, "deadzone"

    if len(z_signal_history) >= 8:
        history = np.array(z_signal_history, dtype=np.float64)
        median_delta = float(np.median(history))
        mad = float(np.median(np.abs(history - median_delta)))
        outlier_limit = max(Z_MAX_DELTA_PER_FRAME, Z_OUTLIER_MAD_FACTOR * mad)

        if abs(z_delta - median_delta) > outlier_limit:
            return 0.0, "hold:outlier"

    z_delta = float(np.clip(
        z_delta,
        -Z_MAX_DELTA_PER_FRAME,
        Z_MAX_DELTA_PER_FRAME
    ))

    z_signal_history.append(z_delta)
    if len(z_signal_history) > Z_HISTORY_SIZE:
        z_signal_history.pop(0)

    return z_delta, "ok"


record_estimated_motion("initial")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Eğer p0 bozulduysa veya nokta sayısı azaldıysa tekrar feature seç
    if p0 is None or len(p0) < min_homography_points:
        p0 = detect_features(old_gray)

        if p0 is None:
            record_estimated_motion("no_features")
            old_gray = frame_gray.copy()
            continue

    # Bir önceki frame'de seçilen feature noktalarını yeni frame'de takip et
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    if p1 is None or st is None:
        record_estimated_motion("optical_flow_failed")
        old_gray = frame_gray.copy()
        p0 = detect_features(old_gray)
        continue

    # Forward-backward kontrolü
    p0_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(
        frame_gray,
        old_gray,
        p1,
        None,
        **lk_params
    )

    if p0_back is None or st_back is None:
        record_estimated_motion("backward_flow_failed")
        old_gray = frame_gray.copy()
        p0 = detect_features(old_gray)
        continue

    fb_error = np.linalg.norm(p0 - p0_back, axis=2)

    valid_flow = (
        (st.ravel() == 1) &
        (st_back.ravel() == 1) &
        (fb_error.ravel() < fb_error_threshold)
    )

    fb_count = int(np.count_nonzero(valid_flow))

    good_old = p0[valid_flow]
    good_new = p1[valid_flow]

    if len(good_new) < min_homography_points:
        record_estimated_motion("low_tracked_points")
        old_gray = frame_gray.copy()
        p0 = detect_features(old_gray)
        continue

    # Noktaları ve optical flow oklarını çiz
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()

        dx = a - c
        dy = b - d

        motion_mag = np.hypot(dx, dy)

        cv2.circle(frame, (int(c), int(d)), 3, point_color, -1)

        if motion_mag < min_motion_threshold:
            continue

        end_x = int(c + dx * arrow_scale)
        end_y = int(d + dy * arrow_scale)

        cv2.arrowedLine(
            frame,
            (int(c), int(d)),
            (end_x, end_y),
            arrow_color,
            arrow_thickness,
            tipLength=arrow_tip_length
        )

    homography_inliers = 0
    clean_inliers = 0

    dx_px = 0.0
    dy_px = 0.0

    dx_h = 0.0
    dy_h = 0.0
    angle_deg = 0.0
    scale = 1.0
    homography_status = "failed"

    affine_depth_signal, affine_image_scale, affine_inliers = affine_depth_from_points(
        good_old,
        good_new,
        ransac_threshold=affine_ransac_threshold
    )

    radial_depth_signal, radial_depth_count = radial_depth_from_points(
        good_old,
        good_new,
        w,
        h
    )

    # Homography hesapla
    H_frame, H_mask = cv2.findHomography(
        good_old,
        good_new,
        cv2.RANSAC,
        ransacReprojThreshold=homography_ransac_threshold
    )

    if H_frame is not None and H_mask is not None and abs(H_frame[2, 2]) > 1e-8:
        homography_inliers = int(np.count_nonzero(H_mask))

        # 1) RANSAC inlier noktalarını al
        h_inliers = H_mask.ravel() == 1
        h_old = good_old[h_inliers]
        h_new = good_new[h_inliers]

        # 2) Homography reprojection error ile ikinci temizlik
        h_old, h_new, h_errors = filter_by_homography_error(
            H_frame,
            h_old,
            h_new,
            max_error=homography_reprojection_error
        )

        clean_inliers = len(h_new)

        if clean_inliers >= min_homography_points:
            # 3) Temiz noktalarla Homography'yi tekrar hesapla
            H_refined, _ = cv2.findHomography(h_old, h_new, 0)

            if H_refined is not None and abs(H_refined[2, 2]) > 1e-8:
                H_frame = H_refined / H_refined[2, 2]

                # 4) Sadece temiz inlier noktalarından median dx/dy hesapla
                inlier_flows = h_new.reshape(-1, 2) - h_old.reshape(-1, 2)

                dx_px = np.median(inlier_flows[:, 0])
                dy_px = np.median(inlier_flows[:, 1])

                # 5) Global Homography biriktir
                global_H = H_frame @ global_H

                if abs(global_H[2, 2]) > 1e-8:
                    global_H = global_H / global_H[2, 2]

                # 6) Bu frame için hareketi 9 örnek nokta üzerinden hesapla
                warped_points = cv2.perspectiveTransform(sample_points, H_frame)
                motion = warped_points.reshape(-1, 2) - sample_points.reshape(-1, 2)

                dx_h = np.median(motion[:, 0])
                dy_h = np.median(motion[:, 1])

                # 7) Global hareketi de 9 örnek nokta üzerinden hesapla
                warped_global = cv2.perspectiveTransform(sample_points, global_H)
                global_motion = warped_global.reshape(-1, 2) - sample_points.reshape(-1, 2)

                global_dx_px = np.median(global_motion[:, 0])
                global_dy_px = np.median(global_motion[:, 1])

                # 8) Anlık açı ve ölçek hesabı
                angle_deg, scale = angle_scale_from_homography(H_frame, w, h)

                # 9) Global açı ve global ölçek hesabı
                global_angle_deg, global_scale = angle_scale_from_homography(global_H, w, h)

                homography_text = (
                    f"H dx:{dx_h:.2f} dy:{dy_h:.2f} "
                    f"ang:{angle_deg:.2f} scale:{scale:.3f} "
                    f"in:{homography_inliers} clean:{clean_inliers}"
                )
                homography_status = "ok"

            else:
                homography_text = "H refine failed"
                homography_status = "refine_failed"

        else:
            homography_text = f"H skipped clean:{clean_inliers}"
            homography_status = "low_clean_inliers"

    else:
        homography_text = "H failed"

    # Nadir irtifada tüm görüntü üniform ölçeklenir; affine benzerlik ölçeği
    # bunun için doğrudan ve tutarlı sinyaldir. Radial yalnızca affine NaN ise devreye girer.
    raw_z_delta = affine_depth_signal
    if not np.isfinite(raw_z_delta):
        raw_z_delta = radial_depth_signal

    z_delta_unscaled, z_update_status = robust_z_delta(
        raw_z_delta,
        homography_status,
        clean_inliers,
        angle_deg
    )

    if z_update_status == "ok":
        filtered_z_delta_unscaled = (
            (1.0 - z_smoothing_alpha) * filtered_z_delta_unscaled +
            z_smoothing_alpha * z_delta_unscaled
        )
    else:
        filtered_z_delta_unscaled = 0.0

    z_position_unscaled += filtered_z_delta_unscaled
    z_text = f"Z unscaled: {z_position_unscaled:.5f} {z_update_status}"

    # FPS hesabı
    curr_time = time.time()
    dt = curr_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = curr_time

    # Ekrana yazdırma
    cv2.putText(
        frame,
        f"dx inlier(px): {dx_px:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"dy inlier(px): {dy_px:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        homography_text,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"global px x:{global_dx_px:.2f} y:{global_dy_px:.2f}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"pts fb:{fb_count} H:{homography_inliers} clean:{clean_inliers}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
    frame,
    f"global ang:{global_angle_deg:.2f} scale:{global_scale:.3f}",
    (20, 245),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 0, 255),
    2
)

<<<<<<< HEAD
    cv2.putText(
        frame,
        z_text,
        (20, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 180),
        2
    )

    cv2.putText(
        frame,
        calibration_text,
        (20, 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    if homography_status == "ok":
        heading_rad = np.deg2rad(map_heading_deg)
        rotated_dx = dx_h * np.cos(heading_rad) - dy_h * np.sin(heading_rad)
        rotated_dy = dx_h * np.sin(heading_rad) + dy_h * np.cos(heading_rad)

        map_x_px += rotated_dx
        map_y_px += rotated_dy

        # Yaw-bias kestirimi için (daha uzun) yaw penceresindeki artışları kaydet.
        if not yaw_bias_done and frame_index <= YAW_CALIBRATION_FRAME_LIMIT:
            motion_increment_samples.append(
                (frame_index, rotated_dx, rotated_dy, heading_update_count)
            )

        # Heading kaynağı: homography açısı. (Essential yaw bu düz/nadir footage'ta dejenere
        # olup ~0 döndürdüğü için kaldırıldı; homography planar sahnede zaten teorik doğru.)
        # Küçük dönüşleri koru: eşik altını silme, EMA ile yumuşatıp her karede biriktir.
        # Yalnızca bariz tek-kare sıçramalarını (glitch) ele.
        # Kestirilen sabit yaw bias'ını her güncellemeden çıkararak drift'i azalt.
        if np.isfinite(angle_deg) and abs(angle_deg) <= yaw_outlier_deg:
            filtered_angle_deg = (
                (1.0 - yaw_smoothing_alpha) * filtered_angle_deg +
                yaw_smoothing_alpha * angle_deg
            )
            map_heading_deg += filtered_angle_deg - yaw_bias_deg
            heading_update_count += 1

        if map_heading_deg > 180.0:
            map_heading_deg -= 360.0
        elif map_heading_deg < -180.0:
            map_heading_deg += 360.0

    record_estimated_motion(homography_status)

    # Yaw penceresi dolunca sabit yaw bias'ını bir kez kestir ve sonrasında uygula.
    # (Daha uzun pencere: küçük sabit bias ancak yeterli birikimle gözlemlenebilir.)
    if (
        not yaw_bias_done and
        frame_index >= YAW_CALIBRATION_FRAME_LIMIT and
        real_motion_by_frame is not None
    ):
        estimated_bias, bias_residual, bias_base = estimate_yaw_bias(
            motion_increment_samples,
            real_motion_by_frame
        )
        # Uzun pencerede gerçek bias belirgin iyileşme verir; gürültüye uydurmamak için
        # yine de küçük bir eşik istiyoruz (>=%1 residual azalması).
        if (
            np.isfinite(bias_residual) and
            np.isfinite(bias_base) and
            bias_residual < bias_base * 0.99
        ):
            yaw_bias_deg = estimated_bias
        yaw_bias_done = True
        print(
            f"Yaw bias: {yaw_bias_deg:.5f} deg/frame "
            f"(residual {bias_base:.2f} -> {bias_residual:.2f} px)"
        )

    real_position = estimate_real_position(
        map_x_px,
        map_y_px,
        float(np.exp(z_position_unscaled)),
        motion_calibration
    )

    trajectory_x, trajectory_y, trajectory_source = select_trajectory_point(
        frame_index,
        real_position,
        map_x_px,
        map_y_px
    )
    trajectory.append((trajectory_x, trajectory_y))

    display_real_position = None
    display_real_source = None

    if (
        real_motion_by_frame is not None and
        frame_index <= CALIBRATION_FRAME_LIMIT and
        frame_index in real_motion_by_frame
    ):
        display_real_position = real_motion_by_frame[frame_index]
        display_real_source = "csv"
    elif real_position is not None:
        display_real_position = real_position
        display_real_source = "scaled"

    if display_real_position is not None:
        # Z güvenilmez kalibre edildiyse (yeterli irtifa açıklığı görülmediyse) belirt.
        z_note = ""
        if (
            display_real_source == "scaled" and
            motion_calibration is not None and
            not motion_calibration.get("z_reliable", False)
        ):
            z_note = "(z?)"
        cv2.putText(
            frame,
            (
                f"real({display_real_source}) "
                f"x:{display_real_position[0]:.2f} "
                f"y:{display_real_position[1]:.2f} "
                f"z:{display_real_position[2]:.2f}{z_note} m"
            ),
            (20, 345),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # --- Canlı trajectory penceresi devre dışı: video sonunda matplotlib karşılaştırma grafiği çiziliyor ---
    """
    traj_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    current_draw_scale = draw_scale

    if len(trajectory) > 1:
        traj_xs = [p[0] for p in trajectory]
        traj_ys = [p[1] for p in trajectory]
        max_abs_x = max(abs(min(traj_xs)), abs(max(traj_xs)), 1.0)
        max_abs_y = max(abs(min(traj_ys)), abs(max(traj_ys)), 1.0)
        usable_half = min(map_w, map_h) * 0.45
        current_draw_scale = min(draw_scale, usable_half / max(max_abs_x, max_abs_y))

    cv2.line(traj_map, (0, map_center_y), (map_w, map_center_y), (80, 80, 80), 1)
    cv2.line(traj_map, (map_center_x, 0), (map_center_x, map_h), (80, 80, 80), 1)

    cv2.circle(traj_map, (map_center_x, map_center_y), 4, (0, 255, 255), -1)
    cv2.putText(
        traj_map,
        "(0,0)",
        (map_center_x + 5, map_center_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1
    )

    for i in range(1, len(trajectory)):
        x1, y1 = trajectory[i - 1]
        x2, y2 = trajectory[i]

        px1 = int(map_center_x + x1 * current_draw_scale)
        py1 = int(map_center_y - y1 * current_draw_scale)
        px2 = int(map_center_x + x2 * current_draw_scale)
        py2 = int(map_center_y - y2 * current_draw_scale)

        if 0 <= px1 < map_w and 0 <= py1 < map_h and 0 <= px2 < map_w and 0 <= py2 < map_h:
            cv2.line(traj_map, (px1, py1), (px2, py2), (0, 0, 255), 2)

    last_x, last_y = trajectory[-1]
    last_px = int(map_center_x + last_x * current_draw_scale)
    last_py = int(map_center_y - last_y * current_draw_scale)

    if 0 <= last_px < map_w and 0 <= last_py < map_h:
        cv2.circle(traj_map, (last_px, last_py), 4, (255, 0, 0), -1)

        heading_rad = np.deg2rad(map_heading_deg)
        arrow_len = 30
        heading_end_x = int(last_px + arrow_len * np.cos(heading_rad))
        heading_end_y = int(last_py - arrow_len * np.sin(heading_rad))
        cv2.arrowedLine(
            traj_map,
            (last_px, last_py),
            (heading_end_x, heading_end_y),
            (0, 255, 0),
            2,
            tipLength=0.35
        )

    cv2.putText(
        traj_map,
        f"{trajectory_source} x:{last_x:.1f} y:{last_y:.1f}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        traj_map,
        f"head:{map_heading_deg:.1f} bias:{yaw_bias_deg:.4f}",
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    """

    cv2.imshow("frame", frame)
    # cv2.imshow("trajectory", traj_map)

    if not windows_positioned:
        cv2.moveWindow("frame", 40, 40)
        # cv2.moveWindow("trajectory", w + 80, 40)
        windows_positioned = True
=======
    cv2.imshow("frame", frame)
>>>>>>> parent of cd5c7ec (dönme mantığında iyileştirmeler yapıldı ve trajectory haritası eklendi)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Bir sonraki frame için referans görüntü ve feature noktalarını güncelle
    old_gray = frame_gray.copy()
    p0 = detect_features(old_gray)

    if p0 is None:
        homography_text = "No features"
        continue


estimated_motion_file.close()
cap.release()
cv2.destroyAllWindows()


def plot_trajectory_comparison():
    """
    Video bittiğinde/durdurulduğunda benim tahminimi (son kalibrasyonla gerçek
    koordinatlara ölçeklenmiş) gerçek CSV hareketiyle aynı eksende karşılaştırır.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib bulunamadı; karşılaştırma grafiği çizilemedi.")
        return

    # Gerçek yol (CSV)
    real_x = []
    real_y = []
    if real_motion_by_frame:
        for frame_no in sorted(real_motion_by_frame):
            position = real_motion_by_frame[frame_no]
            real_x.append(float(position[0]))
            real_y.append(float(position[1]))

    # Tahmin yolu: kalibrasyon varsa gerçek koordinatlara ölçekle, yoksa ham piksel.
    est_x = []
    est_y = []
    for row in estimated_motion_rows:
        if row["status"] not in ("ok", "initial"):
            continue
        if motion_calibration is not None:
            position = estimate_real_position(
                row["map_x_px"],
                row["map_y_px"],
                row["z_unscaled"],
                motion_calibration
            )
            if position is None:
                continue
            est_x.append(float(position[0]))
            est_y.append(float(position[1]))
        else:
            est_x.append(float(row["map_x_px"]))
            est_y.append(float(row["map_y_px"]))

    if not real_x and not est_x:
        print("Karşılaştırma için veri yok.")
        return

    est_unit = "m" if motion_calibration is not None else "px (kalibrasyon yok)"

    plt.figure(figsize=(9, 9))
    if real_x:
        plt.plot(real_x, real_y, "-", color="tab:blue", linewidth=2.0, label="Gerçek hareket (CSV)")
        plt.plot(real_x[0], real_y[0], "o", color="tab:blue", markersize=10, label="Gerçek başlangıç")
        plt.plot(real_x[-1], real_y[-1], "s", color="tab:blue", markersize=9, label="Gerçek bitiş")
    if est_x:
        plt.plot(est_x, est_y, "-", color="tab:red", linewidth=1.5, label=f"Benim tahminim ({est_unit})")
        plt.plot(est_x[0], est_y[0], "o", color="tab:red", markersize=8)
        plt.plot(est_x[-1], est_y[-1], "X", color="tab:red", markersize=12, label="Tahmin bitiş")

    plt.title("Tahmin vs Gerçek Hareket (X-Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    output_path = BASE_DIR / "trajectory_comparison.png"
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    print("Karşılaştırma grafiği kaydedildi:", output_path)
    plt.show()


plot_trajectory_comparison()
