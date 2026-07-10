import  numpy as np
import cv2
import time

from yardımcı_fonksiyonlar_affine import *
from kalibrasyon import load_gt_translations, fit_calibration, calibration_report, apply_calibration

# bize gelen kamera veirlerine gör euygun kamera paremetrelerini seçeceğiz ve ona göre işlem yapacağız.
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


cap = cv2.VideoCapture("/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1.MP4")
#cap = cv2.VideoCapture("/Users/mehmetyilmaz/Desktop/Ornek-Veri-1-RGB.MP4")

lk_params = dict(
    winSize=(21, 21),
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
    print("UYARI: Video çözünürlüğü kayıtlı RGB kalibrasyonlarla birebir veya aynı oranda eşleşmiyor.")
    print("UYARI: En yakın kalibrasyon seçildi; crop/letterbox varsa K, undistortion ve cm hesabı sapabilir.")

# Undistortion haritasını bir kez hesapla
map1, map2 = cv2.initUndistortRectifyMap(
    K, dist_coeffs, None, K, (w, h), cv2.CV_16SC2
)

old_frame = cv2.remap(old_frame, map1, map2, cv2.INTER_LINEAR)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

h, w = old_gray.shape

# Optik eksen (prensip noktası). Ötelemeyi burada değerlendireceğiz:
# yere dik bakan kamerada tam kameranın altındaki yer noktası bu piksele düşer,
# böylece dönme/ölçek etkisi ayrışır ve geriye saf öteleme kalır.
optical_center = np.array([K[0, 2], K[1, 2]], dtype=np.float64)


def affine_reproj_filter(M, old_pts, new_pts, max_err):
    """Benzerlik M'e göre eski noktaları projekte eder; reprojection hatası
    büyük olanları (paralaks/hareketli nesne) eler."""
    old2 = old_pts.reshape(-1, 2).astype(np.float64)
    proj = old2 @ M[:, :2].T + M[:, 2]
    err = np.linalg.norm(proj - new_pts.reshape(-1, 2), axis=1)
    keep = err < max_err
    return old_pts[keep], new_pts[keep]


# Görselleştirme ayarları
arrow_color = (0, 0, 255)
point_color = (0, 255, 255)
arrow_thickness = 2
arrow_tip_length = 0.35
min_motion_threshold = 0.5


# Filtre / Affine (benzerlik) ayarları
min_affine_points = 30
min_track_points = 30
fb_error_threshold = 2.0
affine_ransac_threshold = 1.5    # sıkı: bina paralaksını/aykırıları ele
affine_reproj_error = 1.5        # ikinci temizlik reprojection eşiği (px)

# Düz uçuşta gerçek yaw 0'dır; keyframe başına bu eşiğin altındaki yaw'ı
# gürültü/bias sayıp atıyoruz (yay/drift'i önler). Gerçek dönüşler keyframe
# başına çok daha büyük olduğu için etkilenmez.
yaw_deadband_deg = 0.10

# Görüntünün dönüşü drone'un yaw'ının TERSİDİR. Ekranda dönme hâlâ ters
# görünürse bu işareti +1.0 yap.
YAW_SIGN = -1.0

# -----------------------------------------------------------------------------
# Incremental takip + keyframe'e göre geometri
# -----------------------------------------------------------------------------
# LK'yı KÜÇÜK adımlarla (kare-kare) çalıştırıyoruz -> takip doğru kalır.
# Ama benzerliği her zaman ORİJİNAL keyframe noktalarına göre hesaplıyoruz
# -> baseline büyük -> dönme/ölçek gürültüsü ve drift düşer. Hızlı harekette
# bile keyframe onlarca kare dayanır, yaw entegrasyon adımı çok azalır.
# Keyframe; toplam akış / dönme / ölçek eşiği aşılınca veya nokta azalınca yenilenir.
reanchor_flow_px = 250.0     # keyframe'den bu yana medyan akış eşiği
reanchor_yaw_deg = 15.0      # keyframe'den bu yana yaw eşiği
reanchor_scale_lo = 0.85     # keyframe'e göre ölçek bandı
reanchor_scale_hi = 1.18

# Aktif keyframe: orijinal nokta konumları (sabit referans)
kf_pts0 = detect_features(old_gray)
if kf_pts0 is None:
    raise RuntimeError("İlk frame üzerinde takip edilecek feature bulunamadı.")

prev_gray = old_gray.copy()     # bir önceki kare (incremental LK için)
prev_pts = kf_pts0.copy()       # kf_pts0 ile hizalı, güncel takip konumları

# Keyframe'in dünya pozu (frame0 -> keyframe), iç birimde
kf_heading_deg = 0.0
kf_pos_x = 0.0
kf_pos_y = 0.0
kf_scale = 1.0                  # S = H0 / H_keyframe

# Canlı poz (her kare güncellenir; ekran ve yörünge bunu kullanır)
map_heading_deg = 0.0
map_x_px = 0.0
map_y_px = 0.0
cumulative_scale = 1.0

affine_text = "Affine: waiting"

prev_time = time.time()

# Trajectory haritası ayarları
map_w = 300
map_h = 300
map_center_x = map_w // 2
map_center_y = map_h // 2
draw_scale = 1.0


# -----------------------------------------------------------------------------
# GT (CSV) ile kalibrasyon
# -----------------------------------------------------------------------------
# İlk CALIB_FRAMES frame boyunca CSV değeri DOĞRU kabul edilir; aynı anda kendi
# iç-birim tahminimizle eşleştirme toplanır. CALIB_FRAMES'te iç-birim -> metre
# dönüşümü (ölçek+dönme+yansıma+öteleme) bulunur; sonra kendi tahminimize uygulanır.
# NOT: Bu CSV, çalıştırdığın video ile AYNI uçuşa ait olmalı.
#GT_CSV = "/Users/mehmetyilmaz/Desktop/Ornek-Veri-1-RGB-translation_first450.csv"
GT_CSV = "/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1_translation_first450.csv"
CALIB_FRAMES = 450

gt_x, gt_y, gt_z = load_gt_translations(GT_CSV)
print(f"GT yuklendi: {len(gt_x)} satir, kalibrasyon ilk {CALIB_FRAMES} frame")

frame_idx = 0
calib_M = None
calib_est = []          # kalibrasyon fazında iç-birim tahmin
calib_gt = []           # kalibrasyon fazında metrik GT
calib_info = ""
mode_text = "CALIB(GT)"

disp_x = float(gt_x[0])
disp_y = float(gt_y[0])
traj_disp = [(disp_x, disp_y)]   # ekranda gösterilen metrik yörünge


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Video frame numarası (GT indeksiyle hizalı olması için her karede artar)
    frame_idx += 1

    # Keyframe seti bozulduysa güncel kareye anchor at
    if kf_pts0 is None or len(kf_pts0) < min_affine_points:
        kf_pts0 = detect_features(frame_gray)
        prev_gray = frame_gray.copy()
        prev_pts = None if kf_pts0 is None else kf_pts0.copy()
        kf_heading_deg = map_heading_deg
        kf_pos_x = map_x_px
        kf_pos_y = map_y_px
        kf_scale = cumulative_scale
        continue

    # Incremental takip: prev -> current (küçük, doğru LK adımı)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, frame_gray, prev_pts, None, **lk_params
    )
    if p1 is None or st is None:
        kf_pts0 = None
        prev_gray = frame_gray.copy()
        continue

    # Forward-backward kontrolü (current -> prev)
    p0_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(
        frame_gray, prev_gray, p1, None, **lk_params
    )
    if p0_back is None or st_back is None:
        kf_pts0 = None
        prev_gray = frame_gray.copy()
        continue

    fb_error = np.linalg.norm(prev_pts - p0_back, axis=2)
    alive = (
        (st.ravel() == 1) &
        (st_back.ravel() == 1) &
        (fb_error.ravel() < fb_error_threshold)
    )

    # Hem keyframe referansını hem güncel konumları hizalı tut
    prev_before = prev_pts[alive]    # sadece görselleştirme için (kare-kare akış)
    kf_pts0 = kf_pts0[alive]
    cur_pts = p1[alive]
    tracked = int(len(cur_pts))

    # Bir sonraki kare için referansı güncelle
    prev_gray = frame_gray.copy()
    prev_pts = cur_pts

    # -------------------------------------------------------------------------
    # Benzerlik (similarity) kestirimi: keyframe -> current (BÜYÜK baseline)
    # -------------------------------------------------------------------------
    affine_inliers = 0
    d_yaw = 0.0
    s_rel = 1.0
    flow_mag = 0.0
    affine_status = "failed"
    reanchor = tracked < min_track_points

    if tracked >= min_affine_points:
        M, aff_mask = cv2.estimateAffinePartial2D(
            kf_pts0, cur_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=affine_ransac_threshold
        )

        if M is not None and aff_mask is not None and int(np.count_nonzero(aff_mask)) >= min_affine_points:
            # 1) RANSAC inlier'ları
            inliers = aff_mask.ravel() == 1
            c_kf = kf_pts0[inliers]
            c_cur = cur_pts[inliers]

            # 2) Reprojection ile ikinci temizlik: benzerliğe uymayan noktaları
            #    (bina paralaksı, hareketli nesne) at. Bunlar sahte yaw üretir.
            c_kf, c_cur = affine_reproj_filter(M, c_kf, c_cur, affine_reproj_error)

            # 3) Temiz noktalarla tekrar (daha stabil) fit et
            if len(c_cur) >= min_affine_points:
                M_ref, _ = cv2.estimateAffinePartial2D(c_kf, c_cur, method=cv2.LMEDS)
                if M_ref is not None:
                    M = M_ref

            affine_inliers = len(c_cur)

            A = M[:, :2].astype(np.float64)
            t = M[:, 2].astype(np.float64)

            # Keyframe'e göre görüntü dönüşü ve ölçek
            theta_img = np.arctan2(M[1, 0], M[0, 0])
            s_rel = np.hypot(M[0, 0], M[1, 0])

            # Drone yaw'ı görüntü dönüşünün tersi (YAW_SIGN ile ayarlanabilir)
            d_yaw = YAW_SIGN * np.degrees(theta_img)

            # Düz uçuşta kalan minik yaw bias'ını at (yay'ı önler)
            if abs(d_yaw) < yaw_deadband_deg:
                d_yaw = 0.0

            heading_cur = kf_heading_deg + d_yaw
            S_cur = kf_scale * s_rel

            # Optik eksendeki yer noktasının kayması = yer hareketi; drone tersi
            feature_disp = A @ optical_center + t - optical_center
            drone_disp = -feature_disp

            # İrtifayı normalize et (yükseldikçe px başına daha çok mesafe)
            internal = drone_disp / S_cur

            # Mevcut (yawlanmış) kamera çerçevesinden dünya çerçevesine döndür
            phi = np.deg2rad(heading_cur)
            wx = internal[0] * np.cos(phi) - internal[1] * np.sin(phi)
            wy = internal[0] * np.sin(phi) + internal[1] * np.cos(phi)

            map_heading_deg = heading_cur
            map_x_px = kf_pos_x + wx
            map_y_px = kf_pos_y + wy
            cumulative_scale = S_cur

            affine_status = "ok"
            affine_text = f"dyaw:{d_yaw:.2f} s:{s_rel:.3f} in:{affine_inliers}"

            # Keyframe'den bu yana hareket -> re-anchor kararı
            flows = c_cur.reshape(-1, 2) - c_kf.reshape(-1, 2)
            flow_mag = float(np.median(np.linalg.norm(flows, axis=1)))

            if (flow_mag > reanchor_flow_px or
                    abs(d_yaw) > reanchor_yaw_deg or
                    s_rel < reanchor_scale_lo or s_rel > reanchor_scale_hi):
                reanchor = True
        else:
            affine_text = "Affine failed"
            reanchor = True
    else:
        affine_text = f"low points tracked:{tracked}"
        reanchor = True

    # Kare-kare akış oklarını çiz (uniform aşağı = düz gidiş)
    for new, old in zip(cur_pts, prev_before):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.circle(frame, (int(a), int(b)), 2, point_color, -1)
        if np.hypot(a - c, b - d) >= min_motion_threshold:
            cv2.arrowedLine(
                frame, (int(c), int(d)), (int(a), int(b)),
                arrow_color, arrow_thickness, tipLength=arrow_tip_length
            )

    # -------------------------------------------------------------------------
    # Re-anchor: canlı pozu yeni keyframe pozu olarak kilitle, yeni referans seç
    # -------------------------------------------------------------------------
    if reanchor:
        kf_heading_deg = map_heading_deg
        kf_pos_x = map_x_px
        kf_pos_y = map_y_px
        kf_scale = cumulative_scale
        kf_pts0 = detect_features(frame_gray)
        prev_gray = frame_gray.copy()
        prev_pts = None if kf_pts0 is None else kf_pts0.copy()

    # -------------------------------------------------------------------------
    # GT / kalibrasyon: ilk CALIB_FRAMES frame GT doğru; sonra kendi tahminimiz
    # -------------------------------------------------------------------------
    gt_has = frame_idx < len(gt_x)

    if frame_idx < CALIB_FRAMES and gt_has:
        # Kalibrasyon fazı: CSV değerini doğru say + eşleştirme topla
        disp_x = float(gt_x[frame_idx])
        disp_y = float(gt_y[frame_idx])
        mode_text = "CALIB(GT)"
        if affine_status == "ok":
            calib_est.append([map_x_px, map_y_px])
            calib_gt.append([gt_x[frame_idx], gt_y[frame_idx]])
    else:
        # Handoff: kalibrasyonu bir kez hesapla
        if calib_M is None:
            if len(calib_est) >= 10:
                calib_M, _ = fit_calibration(calib_est, calib_gt)
                rep = calibration_report(calib_M, calib_est, calib_gt)
                print("KALIBRASYON:", rep)
                # Teşhis: kalibrasyon çiftlerini offline analiz için dosyaya dök
                pairs = np.hstack([np.array(calib_est), np.array(calib_gt)])
                np.savetxt("kalibrasyon_pairs.csv", pairs, delimiter=",",
                           header="est_x,est_y,gt_x,gt_y", comments="")
                print("kalibrasyon_pairs.csv yazildi:", len(pairs), "cift")
                calib_info = f"scale:{rep['scale_m_per_unit']:.4f}m/u rmse:{rep['rmse_m']:.2f}m refl:{rep['reflected']}"
            else:
                calib_info = "KALIB: yetersiz nokta"
        # Tahmin fazı: kendi iç-birim pozumuza kalibrasyonu uygula -> metre
        if calib_M is not None:
            disp_x, disp_y = apply_calibration(calib_M, map_x_px, map_y_px)
        mode_text = "ESTIMATE"

    traj_disp.append((disp_x, disp_y))

    # FPS hesabı
    curr_time = time.time()
    dt = curr_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = curr_time

    # Ekrana yazdırma
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, affine_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"[{mode_text}] pos x:{disp_x:.2f} y:{disp_y:.2f} m", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"internal x:{map_x_px:.1f} y:{map_y_px:.1f}", (20, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    cv2.putText(frame, f"heading:{map_heading_deg:.2f} cumScale:{cumulative_scale:.3f}", (20, 171),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, f"track:{tracked} in:{affine_inliers} flow:{flow_mag:.1f}px", (20, 204),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, calib_info, (20, 237),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # -------------------------------------------------------------------------
    # Trajectory haritası
    # -------------------------------------------------------------------------
    traj_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    current_draw_scale = draw_scale

    if len(traj_disp) > 1:
        traj_xs = [p[0] for p in traj_disp]
        traj_ys = [p[1] for p in traj_disp]
        max_abs_x = max(abs(min(traj_xs)), abs(max(traj_xs)), 1.0)
        max_abs_y = max(abs(min(traj_ys)), abs(max(traj_ys)), 1.0)
        usable_half = min(map_w, map_h) * 0.45
        current_draw_scale = min(draw_scale, usable_half / max(max_abs_x, max_abs_y))

    cv2.line(traj_map, (0, map_center_y), (map_w, map_center_y), (80, 80, 80), 1)
    cv2.line(traj_map, (map_center_x, 0), (map_center_x, map_h), (80, 80, 80), 1)

    cv2.circle(traj_map, (map_center_x, map_center_y), 4, (0, 255, 255), -1)
    cv2.putText(traj_map, "(0,0)", (map_center_x + 5, map_center_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    for i in range(1, len(traj_disp)):
        x1, y1 = traj_disp[i - 1]
        x2, y2 = traj_disp[i]

        px1 = int(map_center_x + x1 * current_draw_scale)
        py1 = int(map_center_y - y1 * current_draw_scale)
        px2 = int(map_center_x + x2 * current_draw_scale)
        py2 = int(map_center_y - y2 * current_draw_scale)

        if 0 <= px1 < map_w and 0 <= py1 < map_h and 0 <= px2 < map_w and 0 <= py2 < map_h:
            cv2.line(traj_map, (px1, py1), (px2, py2), (0, 0, 255), 2)

    last_x, last_y = traj_disp[-1]
    last_px = int(map_center_x + last_x * current_draw_scale)
    last_py = int(map_center_y - last_y * current_draw_scale)

    if 0 <= last_px < map_w and 0 <= last_py < map_h:
        cv2.circle(traj_map, (last_px, last_py), 4, (255, 0, 0), -1)

        heading_rad = np.deg2rad(map_heading_deg)
        arrow_len = 30
        heading_end_x = int(last_px + arrow_len * np.cos(heading_rad))
        heading_end_y = int(last_py - arrow_len * np.sin(heading_rad))
        cv2.arrowedLine(traj_map, (last_px, last_py),
                        (heading_end_x, heading_end_y), (0, 255, 0), 2, tipLength=0.35)

    cv2.putText(traj_map, f"x:{last_x:.1f} y:{last_y:.1f} head:{map_heading_deg:.1f}",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("frame", frame)
    cv2.imshow("trajectory", traj_map)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
