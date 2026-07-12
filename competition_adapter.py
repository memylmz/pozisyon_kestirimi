"""Yarışma sunucusu/SDK'sı ile ``TranslationEstimator`` arasındaki ince katman.

Nihai haberleşme paketinin sınıf ve URL adları şartnamede taslak bırakıldığı için
bu modül ağ isteği yapmaz. Yarışma tarafından verilen tahmin nesnesine, ekrandaki
``add_translation_object(DetectedTranslation(...))`` sözleşmesiyle bağlanır.
"""

from __future__ import annotations

from numbers import Integral
from typing import Callable, Optional, Sequence

from translation_estimator import TranslationEstimator, is_healthy


class CompetitionTranslationAdapter:
    """Bir yarışma oturumundaki kareleri tek bir kestirimciyle işler."""

    def __init__(self, estimator: Optional[TranslationEstimator] = None):
        self.estimator = estimator or TranslationEstimator()
        self._last_frame_id = None
        self._last_frame_output = None
        self._last_frame_index = None
        self._session_id = None
        self._video_id = None
        self._last_prediction_object_id = None
        self._last_prediction_key = None

    def reset_session(self) -> None:
        """Her yeni video/oturum başlamadan önce çağrılmalıdır."""
        self.estimator.reset()
        self._last_frame_id = None
        self._last_frame_output = None
        self._last_frame_index = None
        self._session_id = None
        self._video_id = None
        self._last_prediction_object_id = None
        self._last_prediction_key = None

    def _prepare_stream(self, *, session_id=None, video_id=None,
                        frame_id=None, frame_index=None) -> None:
        session_changed = (
            session_id is not None and self._session_id is not None
            and session_id != self._session_id
        )
        video_changed = (
            video_id is not None and self._video_id is not None
            and video_id != self._video_id
        )
        if session_changed or video_changed:
            self.reset_session()
        if session_id is not None:
            self._session_id = session_id
        if video_id is not None:
            self._video_id = video_id
        if frame_index is not None:
            if not isinstance(frame_index, Integral) or isinstance(frame_index, bool):
                raise ValueError("frame_index negatif olmayan bir tam sayı olmalıdır.")
            if frame_index < 0:
                raise ValueError("frame_index negatif olamaz.")
            if self._last_frame_index is not None:
                if frame_index < self._last_frame_index:
                    raise ValueError("Kareler sıra dışı geldi; VO için sıra korunmalıdır.")
                if (frame_index == self._last_frame_index
                        and frame_id != self._last_frame_id):
                    raise ValueError("Aynı frame_index farklı frame_id ile geldi.")
        if (frame_id is not None and frame_id == self._last_frame_id
                and frame_index is not None and self._last_frame_index is not None
                and frame_index != self._last_frame_index):
            raise ValueError("Aynı frame_id farklı frame_index ile geldi.")

    def estimate(self, frame_bgr, health_status: object,
                 translation: Optional[Sequence[float]] = None,
                 frame_id: object = None, frame_index: int = None,
                 session_id: object = None, video_id: object = None):
        """Sağlıklıysa referansı, değilse görüntü kestirimini döndürür."""
        self._prepare_stream(
            session_id=session_id,
            video_id=video_id,
            frame_id=frame_id,
            frame_index=frame_index,
        )
        duplicate_frame = (
            (frame_id is not None and frame_id == self._last_frame_id)
            or (frame_id is None and frame_index is not None
                and frame_index == self._last_frame_index)
        )
        if duplicate_frame and self._last_frame_output is not None:
            return self._last_frame_output
        gt = translation if is_healthy(health_status) else None
        output = self.estimator.process(frame_bgr, health_status, gt=gt)
        if frame_id is not None:
            self._last_frame_id = frame_id
        if frame_id is not None or frame_index is not None:
            self._last_frame_output = output
        if frame_index is not None:
            self._last_frame_index = frame_index
        return output

    def add_to_prediction(
        self,
        prediction,
        detected_translation_cls: Callable[[float, float, float], object],
        frame_bgr,
        health_status: object,
        translation: Optional[Sequence[float]] = None,
        frame_id: object = None,
        frame_index: int = None,
        session_id: object = None,
        video_id: object = None,
    ):
        """Kestirimi ekrandaki yarışma ``prediction`` nesnesine ekler."""
        x, y, z = self.estimate(
            frame_bgr,
            health_status,
            translation,
            frame_id=frame_id,
            frame_index=frame_index,
            session_id=session_id,
            video_id=video_id,
        )
        prediction_key = frame_id if frame_id is not None else frame_index
        if (prediction_key is not None
                and id(prediction) == self._last_prediction_object_id
                and prediction_key == self._last_prediction_key):
            return x, y, z
        prediction.add_translation_object(detected_translation_cls(x, y, z))
        self._last_prediction_object_id = id(prediction)
        self._last_prediction_key = prediction_key
        return x, y, z

    def as_payload(self, frame_bgr, health_status: object,
                   translation: Optional[Sequence[float]] = None,
                   frame_id: object = None, frame_index: int = None,
                   session_id: object = None, video_id: object = None) -> dict:
        """SDK yerine ham JSON hazırlanacaksa kullanılabilecek taslak alanı üretir."""
        x, y, z = self.estimate(
            frame_bgr,
            health_status,
            translation,
            frame_id=frame_id,
            frame_index=frame_index,
            session_id=session_id,
            video_id=video_id,
        )
        return {
            "detected_translations": [{
                "translation_x": x,
                "translation_y": y,
                "translation_z": z,
            }]
        }
