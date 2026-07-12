"""Kare klasörü ve ilk sağlıklı konum CSV'siyle Görev 2 simülasyonu."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from translation_estimator import TranslationEstimator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def parse_frame_number(value, *, fallback=None) -> int:
    """frame_000123, 123 veya 123.0 değerlerinden kare numarası çıkarır."""
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    try:
        number = float(text)
    except ValueError:
        match = re.search(r"(\d+)$", text)
        if match is not None:
            return int(match.group(1))
        if fallback is not None:
            return fallback
        raise ValueError(f"Geçersiz frame numarası: {value!r}") from None
    if np.isfinite(number) and number.is_integer() and number >= 0:
        return int(number)
    raise ValueError(f"Geçersiz frame numarası: {value!r}")


def frame_number_from_path(path: Path, fallback: int) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else fallback


def find_frames(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Kare klasörü bulunamadı: {directory}")
    frames = sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )
    if not frames:
        raise ValueError(f"Klasörde görüntü bulunamadı: {directory}")
    return frames


def load_reference_csv(csv_path: Path) -> dict[int, tuple[float, float, float]]:
    if not csv_path.is_file():
        raise ValueError(f"GT CSV bulunamadı: {csv_path}")
    values = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"translation_x", "translation_y", "translation_z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"GT CSV eksik kolonlar: {', '.join(sorted(missing))}")
        for row_index, row in enumerate(reader):
            row_number = row_index + 2
            try:
                xyz = tuple(float(row[name]) for name in (
                    "translation_x", "translation_y", "translation_z"
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"GT CSV {row_number}. satırı geçersiz.") from exc
            if not np.all(np.isfinite(xyz)):
                raise ValueError(f"GT CSV {row_number}. satırında NaN/Inf var.")
            frame_value = row.get("frame_numbers", "")
            if frame_value:
                try:
                    frame_index = parse_frame_number(frame_value)
                except ValueError as exc:
                    raise ValueError(
                        f"GT CSV {row_number}. satırındaki frame_numbers geçersiz."
                    ) from exc
            else:
                frame_index = row_index
            if frame_index in values:
                raise ValueError(f"GT CSV'de frame {frame_index} birden fazla kez var.")
            values[frame_index] = xyz
    return values


def parse_healthy_ranges(value: str | None):
    """'0:450,550:600' biçimini yarı açık frame aralıklarına çevirir."""
    if value is None:
        return None
    ranges = []
    for item in value.split(","):
        try:
            start_text, end_text = item.strip().split(":", 1)
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise ValueError(
                "healthy_ranges örneğin '0:450,550:600' biçiminde olmalıdır."
            ) from exc
        if start < 0 or end <= start:
            raise ValueError(f"Geçersiz sağlıklı frame aralığı: {item}")
        ranges.append((start, end))
    return ranges


def run_simulation(frames_dir: Path, gt_csv: Path, out_csv: Path, *,
                   healthy_frames: int = 450, max_frames: int | None = None,
                   log_every: int = 50, undistort: bool = True,
                   healthy_ranges=None) -> int:
    if healthy_frames < 1:
        raise ValueError("healthy_frames en az 1 olmalıdır.")
    if out_csv.expanduser().resolve() == gt_csv.expanduser().resolve():
        raise ValueError("Çıktı CSV, GT CSV ile aynı dosya olamaz.")
    frame_paths = find_frames(frames_dir)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames en az 1 olmalıdır.")
        frame_paths = frame_paths[:max_frames]
    references = load_reference_csv(gt_csv)
    ranges = healthy_ranges or [(0, healthy_frames)]
    frame_numbers = [
        frame_number_from_path(path, index)
        for index, path in enumerate(frame_paths)
    ]
    if len(frame_numbers) != len(set(frame_numbers)):
        raise ValueError("Kare dosyalarında yinelenen frame numarası bulundu.")
    healthy_indexes = [
        index for index in range(len(frame_paths))
        if any(start <= index < end for start, end in ranges)
    ]
    missing_references = [
        frame_numbers[index] for index in healthy_indexes
        if frame_numbers[index] not in references
    ]
    if missing_references:
        raise ValueError(
            "Sağlıklı kabul edilen bazı karelerin GT değeri yok. İlk eksik "
            f"frame: {missing_references[0]}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    estimator = TranslationEstimator(undistort=undistort)
    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "translation_x", "translation_y", "translation_z", "frame_numbers",
            "frame_name", "health_status", "position_source",
        ])
        for index, path in enumerate(frame_paths):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Görüntü okunamadı: {path}")
            healthy = any(start <= index < end for start, end in ranges)
            health_status = "1" if healthy else "0"
            frame_number = frame_numbers[index]
            gt = references[frame_number] if healthy else None
            x, y, z = estimator.process(frame, health_status, gt=gt)
            writer.writerow([
                x, y, z, frame_number, path.name, health_status,
                "reference" if healthy else "estimate",
            ])
            processed = index + 1
            if log_every > 0 and (processed % log_every == 0
                                  or processed == len(frame_paths)):
                status = estimator.status
                print(
                    f"{processed}/{len(frame_paths)} | health_status={health_status} | "
                    f"{status.output_mode} | H={'ok' if status.homography_ok else 'failed'} | "
                    f"track={status.tracked_points} inlier={status.homography_inliers} | "
                    f"kalibrasyon={'hazır' if status.calibrated else 'bekliyor'}",
                    flush=True,
                )
    print(f"Sonuç yazıldı: {out_csv}")
    return len(frame_paths)


def parse_args():
    parser = argparse.ArgumentParser(description="Görev 2 yarışma ortamını yerelde simüle eder.")
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--healthy-frames", type=int, default=450)
    parser.add_argument(
        "--healthy-ranges",
        help="Sonradan sağlık dönüşünü de simüle et: 0:450,550:600",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-undistort", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_simulation(
        args.frames, args.gt, args.out,
        healthy_frames=args.healthy_frames,
        max_frames=args.max_frames,
        log_every=args.log_every,
        undistort=not args.no_undistort,
        healthy_ranges=parse_healthy_ranges(args.healthy_ranges),
    )


if __name__ == "__main__":
    main()
