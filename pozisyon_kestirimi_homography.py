import numpy as np
import cv2
import time
import csv

import matplotlib.pyplot as plt

from yardımcı_fonksiyonlar_homography import (
    detect_features, select_camera_calibration, filter_by_homography_error
)
from kalibrasyon import (
    load_gt_translations, fit_calibration, calibration_report,
    apply_calibration, fit_altitude_from_vertical,
    fit_z_calibration, apply_z_calibration, z_calibration_report
)

# -----------------------------------------------------------------------------
# HOMOGRAFİ tabanlı monoküler VO + GT-CSV metrik kalibrasyon.
#
# Fark (affine similarity'ye göre): hareket keyframe -> current arasında
# findHomography ile kestirilir; homografi düzlemsel perspektifi/eğimi doğru
# modeller (RANSAC baskın yer-düzlemini bulur, düzlem-dışı noktaları eler).
# Yaw/ölçek/öteleme, homografiyi OPTİK MERKEZDE lineerleştirerek çıkarılır
# (kırılgan decomposeHomographyMat 4-çözüm belirsizliği YOK). Gerisi affine
# hattıyla aynı: incremental LK, forward-backward, re-anchor'lı keyframe,
# heading/ölçek/konum dead-reckoning ve GT ile tek-seferlik metrik hizalama.
# -----------------------------------------------------------------------------

#cap = cv2.VideoCapture("/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1.MP4")
cap = cv2.VideoCapture("/Users/mehmetyilmaz/Desktop/Ornek-Veri-1-RGB.MP4")

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
    print("UYARI: Video çözünürlüğü kayıtlı kalibrasyonlarla birebir/aynı oranda eşleşmiyor.")
    print("UYARI: En yakın kalibrasyon seçildi; crop/letterbox varsa K ve hesaplar sapabilir.")

# Undistortion haritasını bir kez hesapla
map1, map2 = cv2.initUndistortRectifyMap(
    K, dist_coeffs, None, K, (w, h), cv2.CV_16SC2
)

old_frame = cv2.remap(old_frame, map1, map2, cv2.INTER_LINEAR)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

h, w = old_gray.shape

# Optik eksen (prensip noktası). Homografiyi burada lineerleştireceğiz:
# yere dik bakan kamerada tam altındaki yer noktası bu piksele düşer, böylece
# dönme/ölçek etkisi ayrışır ve geriye saf öteleme kalır.
optical_center = np.array([K[0, 2], K[1, 2]], dtype=np.float64)


def homography_local_motion(H, center):
    """H'yi 'center' (optik eksen) etrafında birinci mertebeden lineerleştirir.

    Döner: (theta_rad, scale, feature_disp[2]) — o noktadaki lokal dönme, ölçek
    ve yer noktasının kayması. decomposeHomographyMat'ın 4-çözüm belirsizliği
    olmadan yaw/ölçek/ötelemeyi tek noktadan verir.
    """
    cx = float(center[0])
    cy = float(center[1])
    wq = H[2, 0] * cx + H[2, 1] * cy + H[2, 2]
    if abs(wq) < 1e-12:
        return 0.0, 1.0, np.zeros(2, dtype=np.float64)
    u = (H[0, 0] * cx + H[0, 1] * cy + H[0, 2]) / wq
    v = (H[1, 0] * cx + H[1, 1] * cy + H[1, 2]) / wq
    # Jacobian (2x2) 'center' noktasında
    dudx = (H[0, 0] - u * H[2, 0]) / wq
    dudy = (H[0, 1] - u * H[2, 1]) / wq
    dvdx = (H[1, 0] - v * H[2, 0]) / wq
    dvdy = (H[1, 1] - v * H[2, 1]) / wq
    # En yakın dönme (polar) ve ölçek (alan karekökü)
    theta = np.arctan2(dvdx - dudy, dudx + dvdy)
    scale = np.sqrt(abs(dudx * dvdy - dudy * dvdx))
    feature_disp = np.array([u - cx, v - cy], dtype=np.float64)
    return float(theta), float(scale), feature_disp


def pointcloud_scale(P, Q):
    """keyframe->current ölçeğini iki EŞLEŞMİŞ nokta bulutunun RMS yarıçap oranından
    kestirir:  s = sqrt( Σ||q-q̄||² / Σ||p-p̄||² ).

    Neden: eskiden ölçek tek noktada (optik merkez) homografi Jacobian'ından
    (sqrt(s_fwd/s_bk)) alınıyordu; bu keyframe başına sistematik ~%0.3 AŞAĞI sapıp
    yüzlerce re-anchor boyunca çarpımsal drift üretiyordu (S 1.0 -> 0.45, yörünge
    ~2.2x şişme). Nokta bulutunun yayılım oranı tüm inlier'ları kullanır; benzerlik
    için yansızdır, dönmeden bağımsızdır ve driften çok daha az etkilenir
    (return-to-altitude ölçek sapması %42 -> %19).
    """
    P = P.reshape(-1, 2).astype(np.float64)
    Q = Q.reshape(-1, 2).astype(np.float64)
    pc = P - P.mean(axis=0)
    qc = Q - Q.mean(axis=0)
    sp2 = float(np.sum(pc ** 2))
    if sp2 < 1e-9:
        return 1.0
    return float(np.sqrt(np.sum(qc ** 2) / sp2))


# Görselleştirme
arrow_color = (0, 0, 255)
point_color = (0, 255, 255)
arrow_thickness = 2
arrow_tip_length = 0.35
min_motion_threshold = 0.5

# Filtre / Homografi ayarları
min_homography_points = 30
min_track_points = 30
fb_error_threshold = 2.0
homography_ransac_threshold = 3.0      # homografi similarity'den daha esnek eşik
homography_reproj_error = 2.5          # ikinci temizlik reprojection eşiği (px)

# Düz uçuşta gerçek yaw 0'dır; keyframe başına bu eşiğin altındaki yaw'ı
# gürültü/bias sayıp atıyoruz.
yaw_deadband_deg = 0.10
# Görüntü dönüşü drone yaw'ının tersidir; ekranda ters görünürse +1.0 yap.
YAW_SIGN = -1.0

# Z (irtifa) işaret konvansiyonu. Kalibrasyon penceresi (ilk CALIB_FRAMES) irtifa
# açısından DÜZ olduğundan (yatay hareket >> dikey), dikey GT kanalından otomatik
# işaret tespiti VO ölçek drifti sinyali bastırırsa yanılabilir. Bu yüzden manuel
# override: 0.0 = otomatik tespit (np.sign(H0_vert)); +1.0/-1.0 = işareti zorla.
# YAW_SIGN ile aynı mantık. Çalıştırınca konsoldaki 'z_isaret' GT'ye göre ters
# görünüyorsa burayı elle ayarla. THYZ_2026_Ornek_Veri_1 GT'si aşağı-pozitif -> -1.0.
Z_SIGN = 0.0

# Keyframe yenileme eşikleri. Değerler gevşetildi: keyframe daha uzun yaşar ->
# daha az keyframe -> kalan ölçek biasının çarpımsal birikim SAYISI azalır
# (headless testte re-anchor 235->142, return-to-altitude drift sapması %19->%16,
# drift şişmesi 1.62x->1.54x). Bias çoğunlukla kat edilen YOLA bağlı olduğundan
# kazanım küçüktür. Aşırı gevşetme baseline'ı büyütüp kenar feature kaybından
# homografiyi zayıflatır; bu değerler bu veride test edilmiş dengedir.
reanchor_flow_px = 450.0
reanchor_yaw_deg = 20.0
reanchor_scale_lo = 0.72
reanchor_scale_hi = 1.35

# Aktif keyframe: orijinal nokta konumları (sabit referans)
kf_pts0 = detect_features(old_gray)
if kf_pts0 is None:
    raise RuntimeError("İlk frame üzerinde takip edilecek feature bulunamadı.")

prev_gray = old_gray.copy()
prev_pts = kf_pts0.copy()

# Keyframe dünya pozu (frame0 -> keyframe), iç birimde
kf_heading_deg = 0.0
kf_pos_x = 0.0
kf_pos_y = 0.0
kf_scale = 1.0                  # S = H0 / H_keyframe

# Canlı poz
map_heading_deg = 0.0
map_x_px = 0.0
map_y_px = 0.0
cumulative_scale = 1.0

homography_text = "Homography: waiting"
prev_time = time.time()

# Trajectory haritası
map_w = 300
map_h = 300
map_center_x = map_w // 2
map_center_y = map_h // 2
draw_scale = 1.0

# -----------------------------------------------------------------------------
# GT (CSV) ile kalibrasyon
# İlk CALIB_FRAMES frame boyunca CSV değeri DOĞRU kabul edilir; aynı anda kendi
# iç-birim tahminimiz toplanır. Handoff'ta iç-birim -> metre dönüşümü bulunur.
# NOT: Bu CSV, çalıştırdığın video ile AYNI uçuşa ait olmalı.
# >>> VERECEĞİN CSV'yi buraya yaz: <<<
# -----------------------------------------------------------------------------
GT_CSV = "/Users/mehmetyilmaz/Desktop/Ornek-Veri-1-RGB-translation_first450.csv"
#GT_CSV = "/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1_translation_first450.csv"
CALIB_FRAMES = 450
OUT_CSV = "/Users/mehmetyilmaz/Desktop/tahmin_translation.csv"

gt_x, gt_y, gt_z = load_gt_translations(GT_CSV)
print(f"GT yuklendi: {len(gt_x)} satir, kalibrasyon ilk {CALIB_FRAMES} frame")

fx = float(K[0, 0])

frame_idx = 0
calib_M = None
calib_est = []
calib_gt = []
calib_scale = []
calib_gz = []
calib_z_raw = []
calib_z_params = None
calib_info = ""
mode_text = "CALIB(GT)"

H0 = None
alt_source = ""

disp_x = float(gt_x[0])
disp_y = float(gt_y[0])
disp_z = float(gt_z[0])
traj_disp = [(disp_x, disp_y)]
hist_frame = [0]
hist_z = [disp_z]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_idx += 1

    # Keyframe seti bozulduysa güncel kareye anchor at
    if kf_pts0 is None or len(kf_pts0) < min_homography_points:
        kf_pts0 = detect_features(frame_gray)
        prev_gray = frame_gray.copy()
        prev_pts = None if kf_pts0 is None else kf_pts0.copy()
        kf_heading_deg = map_heading_deg
        kf_pos_x = map_x_px
        kf_pos_y = map_y_px
        kf_scale = cumulative_scale
        continue

    # Incremental takip: prev -> current
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_pts, None, **lk_params)
    if p1 is None or st is None:
        kf_pts0 = None
        prev_gray = frame_gray.copy()
        continue

    # Forward-backward kontrolü
    p0_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **lk_params)
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

    prev_before = prev_pts[alive]        # sadece görselleştirme (kare-kare akış)
    kf_pts0 = kf_pts0[alive]
    cur_pts = p1[alive]
    tracked = int(len(cur_pts))

    prev_gray = frame_gray.copy()
    prev_pts = cur_pts

    # -------------------------------------------------------------------------
    # Homografi kestirimi: keyframe -> current (BÜYÜK baseline)
    # -------------------------------------------------------------------------
    homography_inliers = 0
    d_yaw = 0.0
    s_rel = 1.0
    flow_mag = 0.0
    homography_status = "failed"
    reanchor = tracked < min_track_points

    if tracked >= min_homography_points:
        H, H_mask = cv2.findHomography(
            kf_pts0, cur_pts, cv2.RANSAC, ransacReprojThreshold=homography_ransac_threshold
        )

        if H is not None and H_mask is not None and abs(H[2, 2]) > 1e-8 \
                and int(np.count_nonzero(H_mask)) >= min_homography_points:
            # 1) RANSAC inlier'ları
            inliers = H_mask.ravel() == 1
            c_kf = kf_pts0[inliers]
            c_cur = cur_pts[inliers]

            # 2) Reprojection ile ikinci temizlik (düzlem-dışı / hareketli nesne)
            c_kf, c_cur, _ = filter_by_homography_error(
                H, c_kf, c_cur, max_error=homography_reproj_error
            )

            # 3) Temiz noktalarla ileri + geri homografi (simetrik bias giderme)
            H_back = None
            if len(c_cur) >= min_homography_points:
                H_ref, _ = cv2.findHomography(c_kf, c_cur, 0)
                if H_ref is not None and abs(H_ref[2, 2]) > 1e-8:
                    H = H_ref / H_ref[2, 2]
                H_bk, _ = cv2.findHomography(c_cur, c_kf, 0)
                if H_bk is not None and abs(H_bk[2, 2]) > 1e-8:
                    H_back = H_bk / H_bk[2, 2]

            homography_inliers = len(c_cur)

            # 4) Optik merkezde lineerleştir -> yaw/ölçek/öteleme AYNI noktadan.
            #    Öteleme (nadir) ve ölçek aynı lineerleştirmeden geldiği için
            #    birbiriyle tutarlı; internal = drone_disp / S tutarlı kalır.
            theta_fwd, s_fwd, fd_fwd = homography_local_motion(H, optical_center)

            # YAW ve ÖTELEME için simetrik (ileri+geri) bias giderme: ideali
            # theta_back=-theta_fwd, fd_back=-fd_fwd; ortalama ortak bias'ı birinci
            # mertebede siler. ÖLÇEK ise artık Jacobian'dan DEĞİL, nokta bulutunun
            # yayılım oranından alınır (aşağıdaki pointcloud_scale) — Jacobian ölçeği
            # keyframe başına sistematik aşağı sapıp drift üretiyordu.
            if H_back is not None:
                theta_bk, s_bk, fd_bk = homography_local_motion(H_back, optical_center)
                theta_img = 0.5 * (theta_fwd - theta_bk)
                feature_disp = 0.5 * (fd_fwd - fd_bk)
            else:
                theta_img = theta_fwd
                feature_disp = fd_fwd
            s_rel = pointcloud_scale(c_kf, c_cur)

            # Drone yaw'ı görüntü dönüşünün tersi
            d_yaw = YAW_SIGN * np.degrees(theta_img)
            if abs(d_yaw) < yaw_deadband_deg:
                d_yaw = 0.0

            heading_cur = kf_heading_deg + d_yaw
            S_cur = kf_scale * s_rel

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

            homography_status = "ok"
            homography_text = f"dyaw:{d_yaw:.2f} s:{s_rel:.3f} in:{homography_inliers}"

            # Keyframe'den bu yana hareket -> re-anchor kararı
            flows = c_cur.reshape(-1, 2) - c_kf.reshape(-1, 2)
            flow_mag = float(np.median(np.linalg.norm(flows, axis=1)))

            if (flow_mag > reanchor_flow_px or
                    abs(d_yaw) > reanchor_yaw_deg or
                    s_rel < reanchor_scale_lo or s_rel > reanchor_scale_hi):
                reanchor = True
        else:
            homography_text = "H failed"
            reanchor = True
    else:
        homography_text = f"low points tracked:{tracked}"
        reanchor = True

    # Kare-kare akış oklarını çiz
    for new, old in zip(cur_pts, prev_before):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.circle(frame, (int(a), int(b)), 2, point_color, -1)
        if np.hypot(a - c, b - d) >= min_motion_threshold:
            cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)),
                            arrow_color, arrow_thickness, tipLength=arrow_tip_length)

    # Re-anchor: canlı pozu yeni keyframe pozu olarak kilitle, yeni referans seç
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
        disp_x = float(gt_x[frame_idx])
        disp_y = float(gt_y[frame_idx])
        disp_z = float(gt_z[frame_idx])
        mode_text = "CALIB(GT)"
        if homography_status == "ok":
            calib_est.append([map_x_px, map_y_px])
            calib_gt.append([gt_x[frame_idx], gt_y[frame_idx]])
            calib_scale.append(cumulative_scale)
            calib_gz.append(gt_z[frame_idx])
            calib_z_raw.append(z_raw)
    else:
        # Handoff: kalibrasyonu bir kez hesapla
        if calib_M is None:
            if len(calib_est) >= 10:
                calib_M, _ = fit_calibration(calib_est, calib_gt)
                rep = calibration_report(calib_M, calib_est, calib_gt)
                print("KALIBRASYON:", rep)

                # --- Irtifa donusum kazanci H0 (ISARETLI) ---
                # disp_z = H0 * (z_raw - 1). H0'in BUYUKLUGU baslangic irtifasi,
                # ISARETI ise GT'nin z konvansiyonunu (yukari+ / asagi+) tasir.
                # Onceki kod abs() aliyordu -> isaret kayboluyor, GT asagi+ ise z
                # ters cikiyordu. Kalibrasyon penceresi irtifa acisindan DUZ oldugu
                # icin (ilk 450 frame yarisma kurali) buyuklugu YATAY kanaldan
                # (guvenilir), isareti DIKEY GT kanalindan aliriz.
                H0_horiz = rep["scale_m_per_unit"] * fx
                H0_vert, u_span = fit_altitude_from_vertical(calib_scale, calib_gz)
                # np.sign(H0_vert): z_raw artarken (irtifa artarken) GT z hangi yone
                # gidiyor? Duz pencerede bile net egilimin isaretini verir. Dikey
                # kanalda hic sinyal yoksa +1 varsayilir (yukari+ konvansiyonu).
                z_sign = float(np.sign(H0_vert)) if (H0_vert is not None and H0_vert != 0) else 1.0
                print(f"IRTIFA: H0_yatay={H0_horiz:.2f}m  H0_dikey={H0_vert}  "
                      f"(u_span={u_span:.4f})  z_isaret={z_sign:+.0f}")

                if H0_vert is not None and u_span > 0.03 and H0_vert != 0:
                    # Dikey kanal iyi kosullu (irtifa penceredE degismis): buyukluk
                    # ve isaret birlikte dikey LS fitinden (abs YOK).
                    H0 = H0_vert
                    alt_source = "vertical(signed)"
                else:
                    # Dikey duz: buyukluk yatay kanaldan, ISARET dikey kanaldan.
                    H0 = z_sign * abs(H0_horiz)
                    alt_source = "horizontal-mag/vertical-sign"

                # Manuel override: Z_SIGN verildiyse buyuklugu koru, isareti zorla.
                if Z_SIGN != 0.0:
                    H0 = float(np.sign(Z_SIGN)) * abs(H0)
                    alt_source += f"+zorla({'+' if Z_SIGN >= 0 else '-'})"

                print(f"IRTIFA secildi: H0={H0:+.2f} "
                      f"(buyukluk={abs(H0):.2f}m isaret={'+' if H0 >= 0 else '-'}) kaynak={alt_source}")

                calib_z_params = fit_z_calibration(calib_z_raw, calib_gz)
                if calib_z_params is not None:
                    z_rep = z_calibration_report(calib_z_params, calib_z_raw, calib_gz)
                    print("Z KALIBRASYON:", z_rep)
                    calib_info = f"xy_rmse:{rep['rmse_m']:.2f}m z_rmse:{z_rep['rmse_m']:.2f}m"
                else:
                    print("Z KALIBRASYON: yetersiz z/scale degisimi -> H0 formulu")
                    calib_info = f"xy_rmse:{rep['rmse_m']:.2f}m z:kalib yok H0:{H0:.1f}m"
            else:
                calib_info = "KALIB: yetersiz nokta"
        # Tahmin fazı: iç-birim pozumuza kalibrasyonu uygula -> metre
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

    # FPS
    curr_time = time.time()
    dt = curr_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = curr_time

    # Ekrana yazdırma
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, homography_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"[{mode_text}] pos x:{disp_x:.2f} y:{disp_y:.2f} z:{disp_z:.2f} m", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"internal x:{map_x_px:.1f} y:{map_y_px:.1f}", (20, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    cv2.putText(frame, f"heading:{map_heading_deg:.2f} cumScale:{cumulative_scale:.3f}", (20, 171),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, f"track:{tracked} in:{homography_inliers} flow:{flow_mag:.1f}px", (20, 204),
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
# Çıktı CSV: her frame için x, y, z (GT ile aynı format)
# İlk CALIB_FRAMES frame = GT; sonrası = tahmin. Atlanan frame olursa son bilinen
# değer ileri taşınır.
# -----------------------------------------------------------------------------
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
# Video bitince: yörünge + irtifa grafiği
# -----------------------------------------------------------------------------
est = np.array(traj_disp, dtype=np.float64)
gt_calib = np.stack([gt_x[:CALIB_FRAMES], gt_y[:CALIB_FRAMES]], axis=1)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].plot(est[:, 0], est[:, 1], "-", color="red", linewidth=1.5, label="Tahmin (metrik)")
axs[0].plot(gt_calib[:, 0], gt_calib[:, 1], "-", color="tab:blue", linewidth=1.5,
            label=f"GT (ilk {CALIB_FRAMES})")
axs[0].scatter([est[0, 0]], [est[0, 1]], c="black", s=40, zorder=5, label="Başlangıç")
axs[0].scatter([est[-1, 0]], [est[-1, 1]], c="green", s=40, zorder=5, label="Son")
axs[0].set_aspect("equal", adjustable="datalim")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")
axs[0].set_title("Homografi - yörünge")
axs[0].grid(True, alpha=0.3)
axs[0].legend()

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
