"""Yarışma için kare-kare çalışan konum kestirim çekirdeği.

Bu sınıf, ``pozisyon_kestirimi_homography.py`` içindeki homografi tabanlı
algoritmayı video/CSV/görselleştirme kodundan ayırır. Algoritmanın hareket
kestirim adımları ve eşikleri aynıdır; tek fark her çağrıda yalnızca bir kare
alıp o kare için sunucuya gönderilecek ``(x, y, z)`` değerini döndürmesidir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from yardımcı_fonksiyonlar_homography import (
    detect_features,
    filter_by_homography_error,
    select_camera_calibration,
)
from kalibrasyon import (
    apply_calibration,
    apply_z_calibration,
    calibration_report,
    fit_altitude_from_vertical,
    fit_calibration,
    fit_z_calibration,
    is_valid_affine,
)


def is_healthy(health_status: object) -> bool:
    """Sunucudan gelebilecek yaygın sağlık gösterimlerini normalize eder."""
    if isinstance(health_status, str):
        value = health_status.strip().lower()
        if value in {"1", "true"}:
            return True
        if value in {"0", "false"}:
            return False
        raise ValueError(f"Geçersiz health_status: {health_status!r}")
    if isinstance(health_status, (bool, np.bool_)):
        return bool(health_status)
    if isinstance(health_status, (int, float, np.integer, np.floating)):
        if health_status == 1:
            return True
        if health_status == 0:
            return False
    raise ValueError(f"Geçersiz health_status: {health_status!r}")


def homography_local_motion(H: np.ndarray, center: np.ndarray):
    """Homografiyi optik merkez çevresinde lineerleştirir."""
    cx, cy = float(center[0]), float(center[1])
    wq = H[2, 0] * cx + H[2, 1] * cy + H[2, 2]
    if abs(wq) < 1e-12:
        return 0.0, 1.0, np.zeros(2, dtype=np.float64)

    u = (H[0, 0] * cx + H[0, 1] * cy + H[0, 2]) / wq
    v = (H[1, 0] * cx + H[1, 1] * cy + H[1, 2]) / wq
    dudx = (H[0, 0] - u * H[2, 0]) / wq
    dudy = (H[0, 1] - u * H[2, 1]) / wq
    dvdx = (H[1, 0] - v * H[2, 0]) / wq
    dvdy = (H[1, 1] - v * H[2, 1]) / wq
    theta = np.arctan2(dvdx - dudy, dudx + dvdy)
    scale = np.sqrt(abs(dudx * dvdy - dudy * dvdx))
    feature_disp = np.array([u - cx, v - cy], dtype=np.float64)
    return float(theta), float(scale), feature_disp


def pointcloud_scale(P: np.ndarray, Q: np.ndarray) -> float:
    """Eşleşmiş iki nokta bulutunun RMS yarıçap oranını hesaplar."""
    P = P.reshape(-1, 2).astype(np.float64)
    Q = Q.reshape(-1, 2).astype(np.float64)
    pc = P - P.mean(axis=0)
    qc = Q - Q.mean(axis=0)
    sp2 = float(np.sum(pc**2))
    if sp2 < 1e-9:
        return 1.0
    return float(np.sqrt(np.sum(qc**2) / sp2))


@dataclass(frozen=True)
class EstimatorStatus:
    frame_index: int
    output_mode: str
    homography_ok: bool
    tracked_points: int
    homography_inliers: int
    calibrated: bool


class TranslationEstimator:
    """Frame-frame monoküler VO ve sağlıklı konumlarla metrik kalibrasyon.

    ``process(frame_bgr, health_status, gt)`` ilk görüntüye göre metre cinsinden
    ``(x, y, z)`` döndürür. Sağlık 1 olduğunda ``gt`` aynen döndürülür ve
    kalibrasyon için biriktirilir. Sağlık 0 olduğunda görüntü tabanlı kestirim
    döndürülür.
    """

    def __init__(self, *, undistort: bool = True, yaw_sign: float = -1.0,
                 z_sign: float = 0.0, camera_matrix=None,
                 dist_coeffs=None):
        self.undistort = undistort
        self.YAW_SIGN = yaw_sign
        self.Z_SIGN = z_sign
        self.camera_matrix = (
            None if camera_matrix is None
            else np.asarray(camera_matrix, dtype=np.float64).copy()
        )
        self.dist_coeffs = (
            None if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1).copy()
        )
        if self.camera_matrix is not None:
            if (self.camera_matrix.shape != (3, 3)
                    or not np.all(np.isfinite(self.camera_matrix))):
                raise ValueError("camera_matrix sonlu 3x3 bir matris olmalıdır.")
            if abs(float(self.camera_matrix[2, 2])) < 1e-12:
                raise ValueError("camera_matrix[2,2] sıfır olamaz.")
        if self.dist_coeffs is not None and not np.all(np.isfinite(self.dist_coeffs)):
            raise ValueError("dist_coeffs yalnızca sonlu değerler içermelidir.")

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.min_homography_points = 30
        self.min_track_points = 30
        self.fb_error_threshold = 2.0
        self.homography_ransac_threshold = 3.0
        self.homography_reproj_error = 2.5
        self.yaw_deadband_deg = 0.10
        self.reanchor_flow_px = 450.0
        self.reanchor_yaw_deg = 20.0
        self.reanchor_scale_lo = 0.72
        self.reanchor_scale_hi = 1.35
        self.absolute_scale_lo = 0.25
        self.absolute_scale_hi = 4.0

        self.initialized = False
        self.frame_index = -1
        self.map1 = self.map2 = None
        self.frame_size = None
        self.fx = None
        self.optical_center = None
        self.prev_gray = self.prev_pts = self.kf_pts0 = None

        self.kf_heading_deg = 0.0
        self.kf_pos_x = self.kf_pos_y = 0.0
        self.kf_scale = 1.0
        self.map_heading_deg = 0.0
        self.map_x_px = self.map_y_px = 0.0
        self.cumulative_scale = 1.0

        self.calib_est = []
        self.calib_gt = []
        self.calib_scale = []
        self.calib_gz = []
        self.calib_z_raw = []
        self.recent_calib_est = []
        self.recent_calib_gt = []
        self.recent_calib_z_raw = []
        self.recent_calib_gz = []
        self.reference_window_size = 120
        self.refit_every = 10
        self.min_initial_calibration_points = 30
        self.previous_healthy = None
        self.has_unhealthy_frame = False
        self.calib_M = None
        self.calib_z_params = None
        self.H0 = None
        # Sağlık daha sonra tekrar 1 olduğunda, mevcut VO-metrik konumu yeni
        # mutlak referansa bağlayan öteleme. Homografi çekirdeğini değiştirmez;
        # yalnızca birikmiş metrik drift'i son sağlıklı karede sıfırlar.
        self.metric_correction = np.zeros(3, dtype=np.float64)
        self.last_reference_correction_frame = None
        self.last_output = (0.0, 0.0, 0.0)
        self.status = EstimatorStatus(-1, "INIT", False, 0, 0, False)

    def reset(self) -> None:
        """Yeni video/oturum için aynı ayarlarla bütün durumu sıfırlar."""
        self.__init__(
            undistort=self.undistort,
            yaw_sign=self.YAW_SIGN,
            z_sign=self.Z_SIGN,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
        )

    @staticmethod
    def _validated_gt(gt: Optional[Sequence[float]]) -> tuple[float, float, float]:
        if gt is None or len(gt) != 3:
            raise ValueError("health_status=1 iken gt=(x, y, z) verilmelidir.")
        values = tuple(float(value) for value in gt)
        if not np.all(np.isfinite(values)):
            raise ValueError("Sağlıklı konum değerleri sonlu sayılar olmalıdır.")
        return values

    def _prepare_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            raise ValueError("frame_bgr geçerli bir OpenCV görüntüsü olmalıdır.")
        if frame_bgr.ndim not in (2, 3):
            raise ValueError(f"Desteklenmeyen görüntü şekli: {frame_bgr.shape}")
        if frame_bgr.size == 0 or frame_bgr.dtype != np.uint8:
            raise ValueError("frame_bgr boş olmayan uint8 bir görüntü olmalıdır.")
        h, w = frame_bgr.shape[:2]
        if self.frame_size is not None and (w, h) != self.frame_size:
            raise ValueError(
                f"Oturum içinde görüntü çözünürlüğü değişti: "
                f"beklenen {self.frame_size[0]}x{self.frame_size[1]}, gelen {w}x{h}."
            )
        if frame_bgr.ndim == 3 and frame_bgr.shape[2] not in (3, 4):
            raise ValueError(f"Desteklenmeyen kanal sayısı: {frame_bgr.shape[2]}")
        frame = frame_bgr
        if self.undistort:
            frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        if frame.ndim == 2:
            return frame
        conversion = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(frame, conversion)

    def _initialize(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        self.frame_size = (w, h)
        if self.camera_matrix is not None:
            K = self.camera_matrix.copy()
            dist = (
                np.zeros(5, dtype=np.float64)
                if self.dist_coeffs is None else self.dist_coeffs
            )
        else:
            selected_calib, sx, sy, same_aspect = select_camera_calibration(w, h)
            if not same_aspect:
                raise ValueError(
                    f"{w}x{h} görüntü için birebir/aynı oranlı kamera "
                    "kalibrasyonu yok. Yarışmada verilen camera_matrix ve "
                    "dist_coeffs değerlerini TranslationEstimator'a verin."
                )
            K = selected_calib["K"].astype(np.float64).copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            dist = selected_calib["dist"]
        self.fx = float(K[0, 0])
        self.optical_center = np.array([K[0, 2], K[1, 2]], dtype=np.float64)
        if self.undistort:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                K, dist, None, K, (w, h), cv2.CV_16SC2
            )
        gray = self._prepare_frame(frame_bgr)
        self.kf_pts0 = detect_features(gray)
        self.prev_gray = gray.copy()
        self.prev_pts = None if self.kf_pts0 is None else self.kf_pts0.copy()
        self.initialized = True

    def _reanchor(self, gray: np.ndarray) -> None:
        self.kf_heading_deg = self.map_heading_deg
        self.kf_pos_x = self.map_x_px
        self.kf_pos_y = self.map_y_px
        self.kf_scale = self.cumulative_scale
        self.kf_pts0 = detect_features(gray)
        self.prev_gray = gray.copy()
        self.prev_pts = None if self.kf_pts0 is None else self.kf_pts0.copy()

    def _track(self, gray: np.ndarray) -> tuple[bool, int, int]:
        if self.kf_pts0 is None or len(self.kf_pts0) < self.min_homography_points \
                or self.prev_pts is None:
            self._reanchor(gray)
            return False, 0, 0

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        if p1 is None or st is None:
            self._reanchor(gray)
            return False, 0, 0
        p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, p1, None, **self.lk_params
        )
        if p0_back is None or st_back is None:
            self._reanchor(gray)
            return False, 0, 0

        fb_error = np.linalg.norm(self.prev_pts - p0_back, axis=2)
        alive = (
            (st.ravel() == 1)
            & (st_back.ravel() == 1)
            & (fb_error.ravel() < self.fb_error_threshold)
        )
        self.kf_pts0 = self.kf_pts0[alive]
        cur_pts = p1[alive]
        tracked = int(len(cur_pts))
        self.prev_gray = gray.copy()
        self.prev_pts = cur_pts

        reanchor = tracked < self.min_track_points
        homography_ok = False
        homography_inliers = 0
        if tracked >= self.min_homography_points:
            H, mask = cv2.findHomography(
                self.kf_pts0,
                cur_pts,
                cv2.RANSAC,
                ransacReprojThreshold=self.homography_ransac_threshold,
            )
            if (H is not None and mask is not None and np.all(np.isfinite(H))
                    and abs(H[2, 2]) > 1e-8
                    and int(np.count_nonzero(mask)) >= self.min_homography_points):
                inliers = mask.ravel() == 1
                c_kf, c_cur, _ = filter_by_homography_error(
                    H,
                    self.kf_pts0[inliers],
                    cur_pts[inliers],
                    max_error=self.homography_reproj_error,
                )
                homography_inliers = len(c_cur)
                if homography_inliers < self.min_homography_points:
                    self._reanchor(gray)
                    return False, tracked, homography_inliers

                H_back = None
                H_ref, _ = cv2.findHomography(c_kf, c_cur, 0)
                if (H_ref is not None and np.all(np.isfinite(H_ref))
                        and abs(H_ref[2, 2]) > 1e-8):
                    H = H_ref / H_ref[2, 2]
                H_bk, _ = cv2.findHomography(c_cur, c_kf, 0)
                if (H_bk is not None and np.all(np.isfinite(H_bk))
                        and abs(H_bk[2, 2]) > 1e-8):
                    H_back = H_bk / H_bk[2, 2]

                theta_fwd, _, fd_fwd = homography_local_motion(H, self.optical_center)
                if H_back is not None:
                    theta_bk, _, fd_bk = homography_local_motion(
                        H_back, self.optical_center
                    )
                    theta_img = 0.5 * (theta_fwd - theta_bk)
                    feature_disp = 0.5 * (fd_fwd - fd_bk)
                else:
                    theta_img, feature_disp = theta_fwd, fd_fwd

                s_rel = pointcloud_scale(c_kf, c_cur)
                motion_values = np.r_[theta_img, feature_disp, s_rel]
                if (not np.all(np.isfinite(motion_values))
                        or not self.absolute_scale_lo <= s_rel <= self.absolute_scale_hi):
                    self._reanchor(gray)
                    return False, tracked, homography_inliers
                d_yaw = self.YAW_SIGN * np.degrees(theta_img)
                if abs(d_yaw) < self.yaw_deadband_deg:
                    d_yaw = 0.0
                heading_cur = self.kf_heading_deg + d_yaw
                scale_cur = self.kf_scale * s_rel
                if (not np.isfinite(scale_cur) or scale_cur <= 1e-9
                        or not np.isfinite(heading_cur)):
                    self._reanchor(gray)
                    return False, tracked, homography_inliers
                internal = -feature_disp / scale_cur
                phi = np.deg2rad(heading_cur)
                wx = internal[0] * np.cos(phi) - internal[1] * np.sin(phi)
                wy = internal[0] * np.sin(phi) + internal[1] * np.cos(phi)
                candidate_state = np.array([
                    heading_cur,
                    self.kf_pos_x + wx,
                    self.kf_pos_y + wy,
                    scale_cur,
                ])
                if not np.all(np.isfinite(candidate_state)):
                    self._reanchor(gray)
                    return False, tracked, homography_inliers
                self.map_heading_deg = heading_cur
                self.map_x_px = float(candidate_state[1])
                self.map_y_px = float(candidate_state[2])
                self.cumulative_scale = scale_cur
                homography_ok = True

                flows = c_cur.reshape(-1, 2) - c_kf.reshape(-1, 2)
                flow_mag = float(np.median(np.linalg.norm(flows, axis=1)))
                reanchor = (
                    flow_mag > self.reanchor_flow_px
                    or abs(d_yaw) > self.reanchor_yaw_deg
                    or s_rel < self.reanchor_scale_lo
                    or s_rel > self.reanchor_scale_hi
                )
            else:
                reanchor = True

        if reanchor:
            self._reanchor(gray)
        return homography_ok, tracked, homography_inliers

    def _add_calibration_sample(self, gt: tuple[float, float, float]) -> None:
        z_raw = 1.0 / max(self.cumulative_scale, 1e-9)
        self.calib_est.append([self.map_x_px, self.map_y_px])
        self.calib_gt.append([gt[0], gt[1]])
        self.calib_scale.append(self.cumulative_scale)
        self.calib_gz.append(gt[2])
        self.calib_z_raw.append(z_raw)
        self.recent_calib_est.append([self.map_x_px, self.map_y_px])
        self.recent_calib_gt.append([gt[0], gt[1]])
        self.recent_calib_z_raw.append(z_raw)
        self.recent_calib_gz.append(gt[2])
        for values in (
            self.recent_calib_est,
            self.recent_calib_gt,
            self.recent_calib_z_raw,
            self.recent_calib_gz,
        ):
            if len(values) > self.reference_window_size:
                del values[:-self.reference_window_size]

    def _reset_recent_reference_window(self) -> None:
        self.recent_calib_est.clear()
        self.recent_calib_gt.clear()
        self.recent_calib_z_raw.clear()
        self.recent_calib_gz.clear()

    def _refit_from_recent_reference(self) -> bool:
        """Geri gelen sağlıklı pencereden yerel yön ve ölçeği yeniden öğrenir."""
        n_points = len(self.recent_calib_est)
        if n_points < 10 or n_points % self.refit_every != 0:
            return False
        M, _ = fit_calibration(self.recent_calib_est, self.recent_calib_gt)
        if not is_valid_affine(M):
            return False
        report = calibration_report(M, self.recent_calib_est, self.recent_calib_gt)
        if not np.all(np.isfinite([
            report["scale_m_per_unit"], report["rmse_m"], report["max_err_m"]
        ])):
            return False
        self.calib_M = M
        self._refit_z_from_recent_reference()
        return True

    def _refit_z_from_recent_reference(self) -> bool:
        local_z = fit_z_calibration(
            self.recent_calib_z_raw, self.recent_calib_gz
        )
        if local_z is None or not np.all(np.isfinite(local_z)):
            return False
        self.calib_z_params = local_z
        return True

    def _fit_metric_calibration(self, *, force: bool = False) -> None:
        if (self.calib_M is not None and not force) or len(self.calib_est) < 10:
            return
        M, _ = fit_calibration(self.calib_est, self.calib_gt)
        if not is_valid_affine(M):
            return
        self.calib_M = M
        report = calibration_report(M, self.calib_est, self.calib_gt)
        h0_horizontal = report["scale_m_per_unit"] * self.fx
        h0_vertical, u_span = fit_altitude_from_vertical(
            self.calib_scale, self.calib_gz
        )
        z_sign = (
            float(np.sign(h0_vertical))
            if h0_vertical is not None and h0_vertical != 0
            else 1.0
        )
        if h0_vertical is not None and u_span > 0.03 and h0_vertical != 0:
            self.H0 = h0_vertical
        else:
            self.H0 = z_sign * abs(h0_horizontal)
        if self.Z_SIGN != 0.0:
            self.H0 = float(np.sign(self.Z_SIGN)) * abs(self.H0)
        self.calib_z_params = fit_z_calibration(self.calib_z_raw, self.calib_gz)

    def _raw_estimated_output(self) -> tuple[float, float, float] | None:
        """Sağlıklı referans düzeltmesi uygulanmamış metrik VO sonucunu döndürür."""
        self._fit_metric_calibration()
        if self.calib_M is None:
            return None
        x, y = apply_calibration(self.calib_M, self.map_x_px, self.map_y_px)
        z_raw = 1.0 / max(self.cumulative_scale, 1e-9)
        if self.calib_z_params is not None:
            z = apply_z_calibration(self.calib_z_params, z_raw)
        elif self.H0 is not None:
            z = self.H0 * (z_raw - 1.0)
        else:
            z = self.last_output[2]
        output = np.asarray([x, y, z], dtype=np.float64)
        if not np.all(np.isfinite(output)):
            return None
        return tuple(float(value) for value in output)

    def _estimated_output(self) -> tuple[float, float, float]:
        raw = self._raw_estimated_output()
        if raw is None:
            # Yarışma ilk bölümde sağlıklı veri vereceği için normalde oluşmaz.
            # Her kareye cevap zorunlu olduğundan geçici takip kaybında son güvenli
            # pozisyonu taşımak, boş/NaN sonuç göndermekten daha güvenlidir.
            return self.last_output
        corrected = np.asarray(raw, dtype=np.float64) + self.metric_correction
        if not np.all(np.isfinite(corrected)):
            return self.last_output
        return tuple(float(value) for value in corrected)

    def _anchor_to_reference(self, gt: tuple[float, float, float]) -> None:
        """VO çıktısını sonradan geri gelen sağlıklı konuma bağlar."""
        raw = self._raw_estimated_output()
        if raw is None:
            return
        self.metric_correction = (
            np.asarray(gt, dtype=np.float64) - np.asarray(raw, dtype=np.float64)
        )
        self.last_reference_correction_frame = self.frame_index

    def process(self, frame_bgr: np.ndarray, health_status: object,
                gt: Optional[Sequence[float]] = None) -> tuple[float, float, float]:
        """Bir kareyi işler ve o kare için gönderilecek konumu döndürür."""
        healthy = is_healthy(health_status)
        healthy_gt = self._validated_gt(gt) if healthy else None
        self.frame_index += 1

        if not self.initialized:
            self._initialize(frame_bgr)
            output = healthy_gt if healthy else self.last_output
            self.last_output = output
            self.previous_healthy = healthy
            self.status = EstimatorStatus(
                self.frame_index, "REFERENCE" if healthy else "ESTIMATE",
                False, 0, 0, False,
            )
            return output

        gray = self._prepare_frame(frame_bgr)
        homography_ok, tracked, inliers = self._track(gray)
        if healthy:
            if self.previous_healthy is False and self.calib_M is not None:
                self._reset_recent_reference_window()
            if homography_ok:
                self._add_calibration_sample(healthy_gt)
                if (self.calib_M is None
                        and len(self.calib_est) >= self.min_initial_calibration_points):
                    self._fit_metric_calibration()
                if self.calib_M is not None:
                    if self.has_unhealthy_frame:
                        self._refit_from_recent_reference()
                    elif len(self.calib_est) % self.refit_every == 0:
                        # İlk kesintisiz sağlıklı bölümde bütün örnekleri kullan;
                        # kısa yerel pencere düz rotada yönü gereksiz bozabilir.
                        self._fit_metric_calibration(force=True)
                        # Z ölçek ilişkisi zamanla değişebildiği için dikey
                        # kanalda bütün uçuş yerine yakın sağlıklı pencereyi kullan.
                        self._refit_z_from_recent_reference()
            # İlk metrik kalibrasyon kurulduktan sonra GPS/konum sağlığı yeniden
            # gelirse, drift'i bu mutlak referansta sıfırla. Sağlıklı kare çıktısı
            # her durumda aşağıda GT'nin kendisi olmaya devam eder.
            if self.calib_M is not None:
                self._anchor_to_reference(healthy_gt)
            output = healthy_gt
            mode = "REFERENCE"
        else:
            output = self._estimated_output()
            mode = "ESTIMATE"
            self.has_unhealthy_frame = True

        self.last_output = output
        self.previous_healthy = healthy
        self.status = EstimatorStatus(
            self.frame_index,
            mode,
            homography_ok,
            tracked,
            inliers,
            self.calib_M is not None,
        )
        return output
