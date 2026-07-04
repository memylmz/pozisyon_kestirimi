import  numpy as np
import cv2
import time

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


#cap = cv2.VideoCapture("/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1.MP4")
cap = cv2.VideoCapture("/Users/mehmetyilmaz/Desktop/2025_HYZ_Ornek_Veriler/Ornek_Veri_Gunduz_Kamera_VO.MP4")

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
    print("UYARI: Video çözünürlüğü kayıtlı RGB kalibrasyonlarla birebir veya aynı oranda eşleşmiyor.")
    print("UYARI: En yakın kalibrasyon seçildi; crop/letterbox varsa K, undistortion ve cm hesabı sapabilir.")

# Undistortion haritasını bir kez hesapla
map1, map2 = cv2.initUndistortRectifyMap(
    K, dist_coeffs, None, K, (w, h), cv2.CV_16SC2
)

old_frame = cv2.remap(old_frame, map1, map2, cv2.INTER_LINEAR)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

h, w = old_gray.shape
grid_size = 60

def create_grid_points():# grid noktaları oluşturmaya yarayan fonksiyon
    points = []
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            cx = x + grid_size / 2
            cy = y + grid_size / 2
            if cx < w and cy < h:
                points.append([cx, cy])
    return np.array(points, dtype=np.float32).reshape(-1, 1, 2)

def reset_points():
    p0 = create_grid_points()
    color = np.random.randint(0, 255, (len(p0), 3))
    return p0, color

# SABİT grid noktaları
p0, color = reset_points() # p0 eski görüntüdeki sabit grid noktasıdır

# -----------------------------
# OK GÖRÜNÜRLÜK AYARLARI
# -----------------------------
arrow_color = (0, 0, 255)      # parlak kırmızı
point_color = (0, 255, 255)    # sarı
arrow_thickness = 3
arrow_tip_length = 0.45
arrow_scale = 3.0
min_motion_threshold = 0.5
min_essential_points = 8
fb_error_threshold = 1.5
affine_ransac_threshold = 3.0

# Essential matrix ile bulunan göreli pozlar ölçek belirsizdir.
# scale_factor gerçek bir ölçümle kalibre edilirse poz birimi anlamlı hale gelir.
scale_factor = 1.0
global_R = np.eye(3, dtype=np.float64)
global_t = np.zeros((3, 1), dtype=np.float64)
pose_text = "Pose: waiting"


prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)# burda görüntüyü piksel kaymalarını düzenliyoruz.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # SABİT GRID noktalarından optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)# bu kısımda eski nokta ile yeni nokta arasındaki px değişimi bulunuyor
    # eski görüntüdeki p0 grid noktaları, yeni görüntüde nereye kaydı
    #st nokta başarıyla takip edildi mi?

    if p1 is None or st is None:
        old_gray = frame_gray.copy()
        continue

    p0_back, st_back, err_back = cv2.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **lk_params)
    if p0_back is None or st_back is None:
        old_gray = frame_gray.copy()
        continue

    fb_error = np.linalg.norm(p0 - p0_back, axis=2)
    valid_flow = (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_error.ravel() < fb_error_threshold)
    fb_count = int(np.count_nonzero(valid_flow))

    good_new = p1[valid_flow]
    good_old = p0[valid_flow]   # sabit grid merkezleri

    if len(good_new) < min_essential_points:
        old_gray = frame_gray.copy()
        continue

    affine_matrix, affine_mask = cv2.estimateAffinePartial2D(
        good_old,
        good_new,
        method=cv2.RANSAC,
        ransacReprojThreshold=affine_ransac_threshold
    )
    if affine_matrix is not None and affine_mask is not None:
        affine_inliers = affine_mask.ravel() == 1
        good_old = good_old[affine_inliers]
        good_new = good_new[affine_inliers]
    affine_count = len(good_new)

    if len(good_new) < min_essential_points:
        old_gray = frame_gray.copy()
        continue

    

    flows = []

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()   # yeni frame'deki konum
        c, d = old.ravel()   # sabit grid noktası

        dx = a - c # iki frame arasındaki piksel cinsinden hareketini bulur.
        dy = b - d
        flows.append([dx, dy])

        motion_mag = np.hypot(dx, dy) # burda hareket vektörünün uzunluğu hesaplanıyor yani pisagor.  iki nokta arasındaki mesafeyi ölçüyor

        # sabit grid noktasını daha belirgin çiz
        cv2.circle(frame, (int(c), int(d)), 3, point_color, -1)

        # çok küçük hareketlerde ok çizme, sadece noktayı göster
        if motion_mag < min_motion_threshold:
            continue

        # sadece görsellik için oku büyüt
        end_x = int(c + dx * arrow_scale)
        end_y = int(d + dy * arrow_scale)

        # oku doğrudan frame üzerine çiz -> daha belirgin görünür
        cv2.arrowedLine(
            frame,
            (int(c), int(d)),
            (end_x, end_y),
            arrow_color,
            arrow_thickness,
            tipLength=arrow_tip_length
        )

    flows = np.array(flows, dtype=np.float32) # flows burada takip edilen noktaların piksel cinsinden hareketlerini tutuyor.
    # burası diziyi numpy dizisine çeviriyor

    # Median optical flow: referans trajectory mantığına göre genel öteleme
    dx_px = np.median(flows[:, 0]) # bütün noktaların dx bunların medianı
    dy_px = np.median(flows[:, 1])# bütün noktaların dy bunların medianı alınıyor ve genel hareket hesaplanıyır

    E, inlier_mask = cv2.findEssentialMat(
        good_old,
        good_new,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    inlier_count = 0
    essential_inliers = 0
    if E is not None and inlier_mask is not None:
        essential_inliers = int(np.count_nonzero(inlier_mask))
        if E.shape != (3, 3):
            E = E[:3, :3]

        _, R, t, pose_mask = cv2.recoverPose(E, good_old, good_new, K, mask=inlier_mask)
        inlier_count = int(np.count_nonzero(pose_mask))

        if inlier_count >= min_essential_points:
            # Yeni göreli pozu dünya/kamera başlangıç koordinatına ekle.
            global_t = global_t + global_R @ (t * scale_factor)
            global_R = R @ global_R

            pose_text = (
                f"t dir x:{t[0, 0]:.2f} y:{t[1, 0]:.2f} z:{t[2, 0]:.2f} "
                f"inliers:{inlier_count}"
            )
        else:
            pose_text = f"Pose skipped: low inliers:{inlier_count}"


    curr_time = time.time()
    dt = curr_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = curr_time

    cv2.putText(frame, f"dx(px): {dx_px:.2f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"dy(px): {dy_px:.2f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, pose_text, (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(frame, f"pos x:{global_t[0, 0]:.2f} y:{global_t[1, 0]:.2f} z:{global_t[2, 0]:.2f}", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(frame, f"pts fb:{fb_count} aff:{affine_count} E:{essential_inliers} pose:{inlier_count}", (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    



    cv2.imshow("frame", frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # sadece referans frame güncellenir, grid noktaları sabit kalır
    old_gray = frame_gray.copy()
   

cap.release()
cv2.destroyAllWindows()