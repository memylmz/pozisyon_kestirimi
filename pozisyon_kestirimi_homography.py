import numpy as np
import cv2
import time


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


def detect_features(gray):
    """
    Görüntü üzerinde takip edilebilir feature/köşe noktalarını bulur.
    """
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )


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


# Global Homography
global_H = np.eye(3, dtype=np.float64)

global_dx_px = 0.0
global_dy_px = 0.0

global_angle_deg = 0.0
global_scale = 1.0

homography_text = "Homography: waiting"

prev_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Eğer p0 bozulduysa veya nokta sayısı azaldıysa tekrar feature seç
    if p0 is None or len(p0) < min_homography_points:
        p0 = detect_features(old_gray)

        if p0 is None:
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

            else:
                homography_text = "H refine failed"

        else:
            homography_text = f"H skipped clean:{clean_inliers}"

    else:
        homography_text = "H failed"

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

    cv2.imshow("frame", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Bir sonraki frame için referans görüntü ve feature noktalarını güncelle
    old_gray = frame_gray.copy()
    p0 = detect_features(old_gray)

    if p0 is None:
        homography_text = "No features"
        continue


cap.release()
cv2.destroyAllWindows()