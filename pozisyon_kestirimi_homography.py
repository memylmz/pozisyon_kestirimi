import numpy as np
import cv2
import time
from yardımcı_fonksiyonlar_homography import *






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

homography_text = "Homography: waiting"
z_text = "Z unscaled: 0.00000"

prev_time = time.time()

map_w = 300
map_h = 300
map_center_x = map_w // 2
map_center_y = map_h // 2
draw_scale = 1.0
trajectory = [(0.0, 0.0)]
map_x_px = 0.0
map_y_px = 0.0
map_heading_deg = 0.0
heading_deadband_deg = 0.1
windows_positioned = False


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

    raw_z_delta = radial_depth_signal
    if not np.isfinite(raw_z_delta):
        raw_z_delta = affine_depth_signal

    if np.isfinite(raw_z_delta) and abs(raw_z_delta) >= z_deadzone:
        z_delta_unscaled = z_direction * raw_z_delta
    else:
        z_delta_unscaled = 0.0

    filtered_z_delta_unscaled = (
        (1.0 - z_smoothing_alpha) * filtered_z_delta_unscaled +
        z_smoothing_alpha * z_delta_unscaled
    )
    z_position_unscaled += filtered_z_delta_unscaled
    z_text = f"Z unscaled: {z_position_unscaled:.5f}"

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

    cv2.putText(
        frame,
        z_text,
        (20, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 180),
        2
    )

    if homography_status == "ok":
        heading_rad = np.deg2rad(map_heading_deg)
        rotated_dx = dx_h * np.cos(heading_rad) - dy_h * np.sin(heading_rad)
        rotated_dy = dx_h * np.sin(heading_rad) + dy_h * np.cos(heading_rad)

        map_x_px += rotated_dx
        map_y_px += rotated_dy

        if abs(angle_deg) > heading_deadband_deg:
            map_heading_deg += angle_deg

        if map_heading_deg > 180.0:
            map_heading_deg -= 360.0
        elif map_heading_deg < -180.0:
            map_heading_deg += 360.0

        trajectory.append((map_x_px, map_y_px))

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
        f"x:{last_x:.1f}px y:{last_y:.1f}px head:{map_heading_deg:.1f}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow("frame", frame)
    cv2.imshow("trajectory", traj_map)

    if not windows_positioned:
        cv2.moveWindow("frame", 40, 40)
        cv2.moveWindow("trajectory", w + 80, 40)
        windows_positioned = True

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Bir sonraki frame için referans görüntü ve feature noktalarını güncelle
    old_gray = frame_gray.copy()
    p0 = detect_features(old_gray)

    if p0 is None:
        homography_text = "No features"
        continue

log_file.close()
cap.release()
cv2.destroyAllWindows()
