import  numpy as np
import cv2
import time
import csv

import matplotlib.pyplot as plt

from yardımcı_fonksiyonlar_affine import (
    detect_features, CALIBRATIONS, select_camera_calibration
)
from kalibrasyon import (
    load_gt_translations, fit_calibration, calibration_report,
    apply_calibration, fit_altitude_from_vertical,
    fit_z_calibration, apply_z_calibration, z_calibration_report
)

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

# Ölçek (irtifa) etkisinin ötelemeye SIZMAMASI için ötelemeyi zoom merkezinde
# (kameranın tam altındaki yer noktası = nadir / genleşme odağı FOE)
# değerlendirmek gerekir. Kamera kusursuz dik ve prensip noktası doğruysa bu
# nokta optik merkezdir; kamerada eğim (pitch/roll) veya kalibrasyon hatası
# varsa nadir noktası optik merkezden kayar. Bu kayma sıfır değilse saf yükselme
# (s != 1) sahte bir x/y ötelemesi olarak sızar: feature_disp += (1-s)*(z-oc).
# Kaymayı (piksel) buraya girersen sızıntı kalkar. (0,0) = eski davranış.
# Doğru değeri elle aramana gerek yok: kalibrasyon sonunda tahmini basılır.
nadir_offset = np.array([0.0, 0.0], dtype=np.float64)   # [dx, dy] piksel
translation_center = optical_center + nadir_offset


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

fx = float(K[0, 0])     # odak uzaklığı (piksel) -> mutlak irtifa için

frame_idx = 0
calib_M = None
calib_est = []          # kalibrasyon fazında iç-birim tahmin
calib_gt = []           # kalibrasyon fazında metrik GT
calib_scale = []        # kalibrasyon fazında cumulative_scale (irtifa oranı)
calib_gz = []           # kalibrasyon fazında GT dikey yer değiştirme (z)
calib_z_raw = []        # kalibrasyon fazında 1/cumulative_scale ham z sinyali
calib_oms = []          # kalibrasyon fazında (1 - s_rel) — nadir-offset teşhisi
calib_fd = []           # kalibrasyon fazında feature_disp xy — nadir-offset teşhisi
calib_z_params = None   # z_metre = a * z_raw + b
calib_info = ""
mode_text = "CALIB(GT)"

# Mutlak irtifa
H0 = None               # başlangıç irtifası (m); kalibrasyonda hesaplanır
alt_source = ""         # "vertical" | "horizontal"

disp_x = float(gt_x[0])
disp_y = float(gt_y[0])
disp_z = float(gt_z[0])
traj_disp = [(disp_x, disp_y)]   # ekranda gösterilen metrik yörünge
hist_frame = [0]                 # her frame video indeksi (grafik x ekseni)
hist_z = [disp_z]                # her frame gösterilen/tahmin edilen z


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
            M_back = None
            if len(c_cur) >= min_affine_points:
                M_ref, _ = cv2.estimateAffinePartial2D(c_kf, c_cur, method=cv2.LMEDS)
                if M_ref is not None:
                    M = M_ref
                # Ters yön (current -> keyframe). En-küçük-kareler, gürültülü
                # kaynak noktalar yüzünden ölçek/dönmeyi sistematik olarak
                # "zayıflatır" (attenuation bias). Bu bias her re-anchor'da
                # çarpımsal birikip drift üretiyor; simetrik kestirimle sileriz.
                M_back, _ = cv2.estimateAffinePartial2D(c_cur, c_kf, method=cv2.LMEDS)

            affine_inliers = len(c_cur)

            A = M[:, :2].astype(np.float64)
            t = M[:, 2].astype(np.float64)

            # Keyframe'e göre görüntü dönüşü ve ölçek (ileri yön)
            theta_fwd = np.arctan2(M[1, 0], M[0, 0])
            s_fwd = np.hypot(M[0, 0], M[1, 0])

            # Simetrik (ileri+geri) bias giderme. İdealde theta_back = -theta_fwd
            # ve s_back = 1/s_fwd olmalı; gerçekte attenuation ikisini de aynı
            # yönde saptırır. Ortalama/geometrik ortalama bu ortak bias'ı birinci
            # mertebede iptal eder -> ölçek ve yaw birikimini (drift) azaltır.
            if M_back is not None:
                theta_back = np.arctan2(M_back[1, 0], M_back[0, 0])
                s_back = np.hypot(M_back[0, 0], M_back[1, 0])
                theta_img = 0.5 * (theta_fwd - theta_back)
                s_rel = np.sqrt(s_fwd / s_back) if s_back > 1e-9 else s_fwd
            else:
                theta_img = theta_fwd
                s_rel = s_fwd

            # Drone yaw'ı görüntü dönüşünün tersi (YAW_SIGN ile ayarlanabilir)
            d_yaw = YAW_SIGN * np.degrees(theta_img)

            # Düz uçuşta kalan minik yaw bias'ını at (yay'ı önler)
            if abs(d_yaw) < yaw_deadband_deg:
                d_yaw = 0.0

            heading_cur = kf_heading_deg + d_yaw
            S_cur = kf_scale * s_rel

            # Optik eksendeki yer noktasının kayması = yer hareketi; drone tersi.
            # Öteleme de dönme/ölçek gibi attenuation/yön bias'ı taşır; bu yüzden
            # theta_img ve s_rel ile AYNI simetrik (ileri+geri) ortalamayı
            # ötelemeye de uygularız -> ortak bias birinci mertebede silinir ve
            # öteleme, yaw/ölçek ile tutarlı olur. M_back yoksa davranış ham
            # ileri kestirimle aynıdır.
            feature_disp = A @ translation_center + t - translation_center
            if M_back is not None:
                A_back = M_back[:, :2].astype(np.float64)
                t_back = M_back[:, 2].astype(np.float64)
                feature_disp_back = A_back @ translation_center + t_back - translation_center
                feature_disp = 0.5 * (feature_disp - feature_disp_back)
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
    z_raw = 1.0 / max(cumulative_scale, 1e-9)

    if frame_idx < CALIB_FRAMES and gt_has:
        # Kalibrasyon fazı: CSV değerini doğru say + eşleştirme topla
        disp_x = float(gt_x[frame_idx])
        disp_y = float(gt_y[frame_idx])
        disp_z = float(gt_z[frame_idx])
        mode_text = "CALIB(GT)"
        if affine_status == "ok":
            calib_est.append([map_x_px, map_y_px])
            calib_gt.append([gt_x[frame_idx], gt_y[frame_idx]])
            calib_scale.append(cumulative_scale)
            calib_gz.append(gt_z[frame_idx])
            calib_z_raw.append(z_raw)
            calib_oms.append(1.0 - s_rel)
            calib_fd.append([float(feature_disp[0]), float(feature_disp[1])])
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

                # Nadir-offset (zoom merkezi) teşhisi. Ölçek etkisi ötelemeye
                # sızıyorsa (yükselirken x/y kayması) feature_disp ~ d + (1-s)*(z-oc)
                # olur; (1-s)'e regresyon eğimi ~ (z-oc) = nadir kayması (px).
                # Basılan değeri yukarıdaki nadir_offset'e yazıp yeniden
                # çalıştırınca sızıntı azalır; tekrar ~0 basması kalibre olduğunu
                # gösterir. (d ile (1-s) korelasyonsuz VE kalibrasyon penceresinde
                # irtifa yeterince değişmişse anlamlıdır; değilse "yetersiz" der.)
                oms = np.asarray(calib_oms, dtype=np.float64)
                fd = np.asarray(calib_fd, dtype=np.float64)
                denom_o = float(oms @ oms)
                oms_span = float(oms.max() - oms.min()) if len(oms) else 0.0
                z_off = (oms @ fd) / denom_o if denom_o > 1e-9 else np.zeros(2)
                # Güvenilirlik iki koşula bağlı: (a) kalibrasyon penceresinde
                # ölçek yeterince değişmeli (yoksa (1-s)'e regresyon patlar ve
                # görüntü dışına düşen anlamsız devasa bir offset üretir), (b)
                # sonuç fiziksel olarak görüntü içinde kalmalı. Biri sağlanmazsa
                # tahmin gürültüdür; kullanıcıyı yanlış (büyük) değere karşı uyar.
                plausible = abs(z_off[0]) < w and abs(z_off[1]) < h
                if len(oms) >= 10 and oms_span > 0.05 and plausible:
                    print(f"NADIR-OFFSET tahmini (px): dx={z_off[0]:+.1f} dy={z_off[1]:+.1f}"
                          f"  -> nadir_offset'e yaz")
                else:
                    print(f"NADIR-OFFSET tahmini: guvenilmez (kalibrasyonda irtifa "
                          f"~sabit, olcek span={oms_span:.4f}); nadir_offset=(0,0) birak")

                # --- Mutlak irtifa H0 ---
                # 1) Yatay ölçekten: scale_cal = GSD0 = H0/f  ->  H0 = scale_cal * f
                H0_horiz = rep["scale_m_per_unit"] * fx
                # 2) Dikey kanaldan (yatay dejenerasyondan bağımsız)
                H0_vert, u_span = fit_altitude_from_vertical(calib_scale, calib_gz)
                print(f"IRTIFA: H0_yatay={H0_horiz:.2f}m  H0_dikey={H0_vert}  (u_span={u_span:.4f})")

                # İrtifa penceresinde yeterli değişim varsa dikey kanalı tercih et
                if H0_vert is not None and u_span > 0.03 and H0_vert != 0:
                    H0 = abs(H0_vert)
                    alt_source = "vertical"
                else:
                    H0 = abs(H0_horiz)
                    alt_source = "horizontal"
                print(f"IRTIFA secildi: H0={H0:.2f}m kaynak={alt_source}")

                calib_z_params = fit_z_calibration(calib_z_raw, calib_gz)
                if calib_z_params is not None:
                    z_rep = z_calibration_report(calib_z_params, calib_z_raw, calib_gz)
                    print("Z KALIBRASYON:", z_rep)
                    calib_info = (
                        f"xy_rmse:{rep['rmse_m']:.2f}m "
                        f"z_rmse:{z_rep['rmse_m']:.2f}m"
                    )
                else:
                    print("Z KALIBRASYON: yetersiz z/scale degisimi")
                    calib_info = f"xy_rmse:{rep['rmse_m']:.2f}m z:kalib yok H0:{H0:.1f}m"
            else:
                calib_info = "KALIB: yetersiz nokta"
        # Tahmin fazı: kendi iç-birim pozumuza kalibrasyonu uygula -> metre
        if calib_M is not None:
            disp_x, disp_y = apply_calibration(calib_M, map_x_px, map_y_px)
        if calib_z_params is not None:
            disp_z = apply_z_calibration(calib_z_params, z_raw)
        elif H0 is not None:
            disp_z = H0 * (z_raw - 1.0)
        mode_text = "ESTIMATE"

    traj_disp.append((disp_x, disp_y))
    hist_frame.append(frame_idx)
    hist_z.append(disp_z)

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
    cv2.putText(frame, f"[{mode_text}] pos x:{disp_x:.2f} y:{disp_y:.2f} z:{disp_z:.2f} m", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"internal x:{map_x_px:.1f} y:{map_y_px:.1f}", (20, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    cv2.putText(frame, f"heading:{map_heading_deg:.2f} cumScale:{cumulative_scale:.3f}", (20, 171),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, f"track:{tracked} in:{affine_inliers} flow:{flow_mag:.1f}px", (20, 204),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, calib_info, (20, 237),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    if calib_z_params is not None:
        z_txt = f"Z tahmin: {disp_z:.2f} m  raw:{z_raw:.4f}"
    else:
        z_txt = "Z tahmin: kalibrasyon bekleniyor"
    cv2.putText(frame, z_txt, (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

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


# -----------------------------------------------------------------------------
# Çıktı CSV: her frame için x, y, z değişimi (GT ile aynı format)
# İlk CALIB_FRAMES frame = GT; sonrası = kendi tahminimiz. z = irtifa değişimi.
# Atlanan frame olursa son bilinen değer ileri taşınır (her frame dolu olsun).
# -----------------------------------------------------------------------------
OUT_CSV = "/Users/mehmetyilmaz/Desktop/tahmin_translation.csv"

frame_data = {}
for i in range(len(traj_disp)):
    f = hist_frame[i]
    x, y = traj_disp[i]
    z = float(hist_z[i])
    frame_data[f] = (float(x), float(y), z)

max_f = max(frame_data) if frame_data else 0
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["translation_x", "translation_y", "translation_z", "frame_numbers"])
    last = (0.0, 0.0, 0.0)
    for fi in range(max_f + 1):
        if fi in frame_data:
            last = frame_data[fi]
        x, y, z = last
        wr.writerow([x, y, z, fi])
print(f"{OUT_CSV} yazildi: {max_f + 1} frame")


# -----------------------------------------------------------------------------
# Video bitince: gerçek hareket (yörünge) + irtifa grafiği
# -----------------------------------------------------------------------------
est = np.array(traj_disp, dtype=np.float64)
gt_calib = np.stack([gt_x[:CALIB_FRAMES], gt_y[:CALIB_FRAMES]], axis=1)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 1) x-y yörünge (metrik)
axs[0].plot(est[:, 0], est[:, 1], "-", color="red", linewidth=1.5, label="Tahmin (metrik)")
axs[0].plot(gt_calib[:, 0], gt_calib[:, 1], "-", color="tab:blue", linewidth=1.5,
            label=f"GT (ilk {CALIB_FRAMES})")
axs[0].scatter([est[0, 0]], [est[0, 1]], c="black", s=40, zorder=5, label="Başlangıç")
axs[0].scatter([est[-1, 0]], [est[-1, 1]], c="green", s=40, zorder=5, label="Son")
axs[0].set_aspect("equal", adjustable="datalim")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")
axs[0].set_title("Gerçek hareket - yörünge")
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# 2) z (m) - frame
axs[1].plot(hist_frame, hist_z, "-", color="tab:green", linewidth=1.5, label="Tahmin/ekran z")
gt_plot_n = min(len(gt_z), max(hist_frame) + 1)
if gt_plot_n > 0:
    axs[1].plot(range(gt_plot_n), gt_z[:gt_plot_n], "--", color="tab:blue", linewidth=1.2, label="GT z")
axs[1].axvline(CALIB_FRAMES, color="gray", linestyle="--", alpha=0.7, label="kalibrasyon sonu")
axs[1].set_title("Z tahmini")
axs[1].legend()
axs[1].set_xlabel("frame")
axs[1].set_ylabel("z (m)")
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
