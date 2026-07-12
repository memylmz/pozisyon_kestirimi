"""Yarışma konum kestirimindeki kritik güvenlik regresyonları."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from competition_adapter import CompetitionTranslationAdapter
from kalibrasyon import fit_calibration, is_valid_affine
from translation_estimator import TranslationEstimator, is_healthy
from yarisma_simulasyonu import parse_frame_number, parse_healthy_ranges


class CompetitionEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.base = cv2.GaussianBlur(
            rng.integers(0, 256, (512, 640, 3), dtype=np.uint8), (5, 5), 0
        )
        cls.shifted = cv2.warpAffine(
            cls.base,
            np.float32([[1, 0, 4], [0, 1, 2]]),
            (640, 512),
            borderMode=cv2.BORDER_REFLECT,
        )

    def test_collinear_calibration_has_similarity_fallback(self):
        x = np.arange(30, dtype=np.float32)
        estimated = np.column_stack([x, np.zeros_like(x)])
        reference = np.column_stack([2 * x, np.zeros_like(x)])
        matrix, _ = fit_calibration(estimated, reference)
        self.assertTrue(is_valid_affine(matrix))
        self.assertFalse(is_valid_affine(np.array([
            [1.0, 0.0, 0.0], [0.0, 1e-12, 0.0]
        ])))

    def test_empty_clean_inliers_are_rejected(self):
        estimator = TranslationEstimator(undistort=False)
        estimator.process(self.base, "1", (0.0, 0.0, 0.0))

        def empty_filter(H, old_points, new_points, max_error):
            return old_points[:0], new_points[:0], np.empty(0)

        with patch("translation_estimator.filter_by_homography_error", empty_filter):
            estimator.process(self.shifted, "1", (1.0, 1.0, 0.0))
        self.assertFalse(estimator.status.homography_ok)
        self.assertTrue(np.isfinite([
            estimator.map_x_px, estimator.map_y_px, estimator.cumulative_scale
        ]).all())

    def test_zero_scale_is_rejected(self):
        estimator = TranslationEstimator(undistort=False)
        estimator.process(self.base, "1", (0.0, 0.0, 0.0))
        with patch("translation_estimator.pointcloud_scale", return_value=0.0):
            estimator.process(self.shifted, "1", (1.0, 1.0, 0.0))
        self.assertFalse(estimator.status.homography_ok)
        self.assertTrue(np.isfinite([
            estimator.map_x_px, estimator.map_y_px, estimator.cumulative_scale
        ]).all())

    def test_duplicate_frame_id_does_not_advance_estimator(self):
        adapter = CompetitionTranslationAdapter(
            TranslationEstimator(undistort=False)
        )
        first = adapter.estimate(
            self.base, "1", (0.0, 0.0, 0.0), frame_id="frame-0"
        )
        frame_index = adapter.estimator.frame_index
        second = adapter.estimate(
            self.base, "1", (0.0, 0.0, 0.0), frame_id="frame-0"
        )
        self.assertEqual(first, second)
        self.assertEqual(adapter.estimator.frame_index, frame_index)

        class Translation:
            def __init__(self, x, y, z):
                self.xyz = (x, y, z)

        class Prediction:
            def __init__(self):
                self.items = []

            def add_translation_object(self, value):
                self.items.append(value)

        adapter.reset_session()
        prediction = Prediction()
        kwargs = dict(
            prediction=prediction,
            detected_translation_cls=Translation,
            frame_bgr=self.base,
            health_status="1",
            translation=(0.0, 0.0, 0.0),
            frame_id="frame-0",
            frame_index=0,
        )
        adapter.add_to_prediction(**kwargs)
        adapter.add_to_prediction(**kwargs)
        self.assertEqual(len(prediction.items), 1)
        payload = adapter.as_payload(
            self.base, "1", (0.0, 0.0, 0.0), frame_id="frame-0", frame_index=0
        )
        self.assertEqual(len(payload["detected_translations"]), 1)

    def test_dynamic_health_ranges(self):
        self.assertEqual(
            parse_healthy_ranges("0:450,550:600"),
            [(0, 450), (550, 600)],
        )
        self.assertEqual(parse_frame_number("frame_000123"), 123)
        self.assertEqual(parse_frame_number("123.0"), 123)

    def test_invalid_health_and_changed_resolution_are_rejected(self):
        with self.assertRaises(ValueError):
            is_healthy("unknown")
        estimator = TranslationEstimator(undistort=False)
        estimator.process(self.base, "1", (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            estimator.process(self.base[:500], "0")

    def test_session_change_resets_and_out_of_order_is_rejected(self):
        adapter = CompetitionTranslationAdapter(
            TranslationEstimator(undistort=False)
        )
        adapter.estimate(
            self.base, "1", (0.0, 0.0, 0.0),
            frame_id="a-0", frame_index=0, session_id="a", video_id="v1",
        )
        adapter.estimate(
            self.shifted, "1", (1.0, 1.0, 0.0),
            frame_id="a-1", frame_index=1, session_id="a", video_id="v1",
        )
        with self.assertRaises(ValueError):
            adapter.estimate(
                self.base, "1", (0.0, 0.0, 0.0),
                frame_id="late", frame_index=0, session_id="a", video_id="v1",
            )
        adapter.estimate(
            self.base, "1", (0.0, 0.0, 0.0),
            frame_id="b-0", frame_index=0, session_id="b", video_id="v2",
        )
        self.assertEqual(adapter.estimator.frame_index, 0)


if __name__ == "__main__":
    unittest.main()
