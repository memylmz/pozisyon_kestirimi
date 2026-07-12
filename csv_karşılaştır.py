import argparse
import csv
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# IDE'den (argümansız "Run") çalıştırmak için varsayılan iki CSV yolu.
# Buraya karşılaştırmak istediğin iki dosyayı yaz; ya da CLI'dan geç:
#   python csv_karsilastir.py birinci.csv ikinci.csv
# CSV_1 genelde GERÇEK (GT), CSV_2 TAHMİN olarak düşünülür (etiketler ona göre).
# -----------------------------------------------------------------------------
DEFAULT_CSV_1 = "/Users/mehmetyilmaz/Desktop/Ornek-Veri-1-RGB_tahmin_translation.csv"
#DEFAULT_CSV_1 = "/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1_translation.csv"
DEFAULT_CSV_2 = "/Users/mehmetyilmaz/Desktop/tahmin_translation.csv"

# Eşleşen frame'ler arasında kaç frame'de bir sapma (divergence) çizgisi çizilsin.
# Yörüngeyi boğmamak için seyrek çiz. 0 -> hiç çizme.
DIVERGENCE_EVERY = 200


X_COLUMN_CANDIDATES = (
    "translation_x",
    "x",
    "pos_x",
    "position_x",
    "map_x",
    "est_x",
    "gt_x",
)
Y_COLUMN_CANDIDATES = (
    "translation_y",
    "y",
    "pos_y",
    "position_y",
    "map_y",
    "est_y",
    "gt_y",
)
FRAME_COLUMN_CANDIDATES = ("frame_numbers", "frame", "frame_idx", "index")


def find_column(fieldnames, candidates):
    normalized = {name.lower().strip(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_xy_csv(csv_path, x_col=None, y_col=None, frame_col=None):
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} icinde baslik satiri bulunamadi.")

        fieldnames = reader.fieldnames
        x_col = x_col or find_column(fieldnames, X_COLUMN_CANDIDATES)
        y_col = y_col or find_column(fieldnames, Y_COLUMN_CANDIDATES)
        frame_col = frame_col or find_column(fieldnames, FRAME_COLUMN_CANDIDATES)

        if x_col is None or y_col is None:
            raise ValueError(
                f"{path} icin x/y kolonlari bulunamadi. "
                f"Kolonlar: {', '.join(fieldnames)}"
            )

        frames = []
        xs = []
        ys = []
        for row_index, row in enumerate(reader):
            try:
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
                if frame_col is not None and row.get(frame_col, "") != "":
                    frames.append(int(float(row[frame_col])))
                else:
                    frames.append(row_index)
            except ValueError:
                continue

    if not xs:
        raise ValueError(f"{path} icinden sayisal x/y verisi okunamadi.")

    return {
        "path": path,
        "x_col": x_col,
        "y_col": y_col,
        "frame_col": frame_col,
        "frames": np.asarray(frames, dtype=np.int64),
        "xy": np.column_stack([xs, ys]).astype(np.float64),
    }


def compare_by_common_frames(first, second):
    first_by_frame = {frame: xy for frame, xy in zip(first["frames"], first["xy"])}
    second_by_frame = {frame: xy for frame, xy in zip(second["frames"], second["xy"])}
    common_frames = sorted(set(first_by_frame).intersection(second_by_frame))

    if common_frames:
        first_xy = np.asarray([first_by_frame[frame] for frame in common_frames])
        second_xy = np.asarray([second_by_frame[frame] for frame in common_frames])
        return np.asarray(common_frames), first_xy, second_xy

    count = min(len(first["xy"]), len(second["xy"]))
    return np.arange(count), first["xy"][:count], second["xy"][:count]


def error_report(first_xy, second_xy):
    diff = first_xy - second_xy
    err = np.linalg.norm(diff, axis=1)
    return {
        "n": int(len(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "max": float(np.max(err)),
        "final": float(err[-1]),
        "mean_dx": float(np.mean(diff[:, 0])),
        "mean_dy": float(np.mean(diff[:, 1])),
    }


def plot_trajectories(first, second, report, common=None, output_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 7))

    # Eşleşen frame'ler arasında sapma çizgileri (aynı frame -> nereye kaymış).
    # Yörünge çizgilerinin ALTINDA kalsın diye önce çiziyoruz.
    if common is not None and DIVERGENCE_EVERY > 0:
        cf, fxy, sxy = common
        step = max(1, DIVERGENCE_EVERY)
        for i in range(0, len(cf), step):
            plt.plot(
                [fxy[i, 0], sxy[i, 0]],
                [fxy[i, 1], sxy[i, 1]],
                color="gray",
                linewidth=0.6,
                alpha=0.5,
                zorder=1,
            )

    plt.plot(
        first["xy"][:, 0],
        first["xy"][:, 1],
        color="tab:blue",
        linewidth=2,
        label=f"1) {first['path'].stem}",
        zorder=2,
    )
    plt.plot(
        second["xy"][:, 0],
        second["xy"][:, 1],
        color="tab:orange",
        linewidth=2,
        label=f"2) {second['path'].stem}",
        zorder=2,
    )

    plt.scatter(first["xy"][0, 0], first["xy"][0, 1], color="tab:blue", marker="o", s=60,
                zorder=3, label="Başlangıç")
    plt.scatter(first["xy"][-1, 0], first["xy"][-1, 1], color="tab:blue", marker="x", s=90,
                zorder=3, label="Bitiş")
    plt.scatter(second["xy"][0, 0], second["xy"][0, 1], color="tab:orange", marker="o", s=60, zorder=3)
    plt.scatter(second["xy"][-1, 0], second["xy"][-1, 1], color="tab:orange", marker="x", s=90, zorder=3)

    plt.title(
        "CSV Yörünge Karşılaştırması\n"
        f"RMSE: {report['rmse']:.3f} m | Ort: {report['mean']:.3f} m | "
        f"Max: {report['max']:.3f} m | Son nokta hatası: {report['final']:.3f} m"
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=180)
        print(f"Grafik kaydedildi: {output_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iki CSV dosyasindaki x-y konumlarini ayni grafikte karsilastirir."
    )
    parser.add_argument("csv_1", nargs="?", default=DEFAULT_CSV_1,
                        help=f"Birinci CSV (varsayilan: {DEFAULT_CSV_1})")
    parser.add_argument("csv_2", nargs="?", default=DEFAULT_CSV_2,
                        help=f"Ikinci CSV (varsayilan: {DEFAULT_CSV_2})")
    parser.add_argument("--x1", help="Birinci CSV icin x kolonu")
    parser.add_argument("--y1", help="Birinci CSV icin y kolonu")
    parser.add_argument("--x2", help="Ikinci CSV icin x kolonu")
    parser.add_argument("--y2", help="Ikinci CSV icin y kolonu")
    parser.add_argument("--frame1", help="Birinci CSV icin frame kolonu")
    parser.add_argument("--frame2", help="Ikinci CSV icin frame kolonu")
    parser.add_argument(
        "--out",
        help="Grafiği ekranda gostermek yerine verilen dosyaya kaydet. Ornek: karsilastirma.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    first = load_xy_csv(args.csv_1, args.x1, args.y1, args.frame1)
    second = load_xy_csv(args.csv_2, args.x2, args.y2, args.frame2)

    common_frames, first_common, second_common = compare_by_common_frames(first, second)
    report = error_report(first_common, second_common)

    print("Okunan kolonlar:")
    print(f"  1) {first['path']}: x={first['x_col']} y={first['y_col']} frame={first['frame_col']}")
    print(f"  2) {second['path']}: x={second['x_col']} y={second['y_col']} frame={second['frame_col']}")
    print("Karsilastirma:")
    print(f"  Ortak nokta sayisi: {report['n']}")
    print(f"  RMSE: {report['rmse']:.3f} m")
    print(f"  Ortalama hata: {report['mean']:.3f} m")
    print(f"  Medyan hata: {report['median']:.3f} m")
    print(f"  Maksimum hata: {report['max']:.3f} m")
    print(f"  Son nokta hatasi: {report['final']:.3f} m")
    print(f"  Ortalama dx: {report['mean_dx']:.3f} m")
    print(f"  Ortalama dy: {report['mean_dy']:.3f} m")

    plot_trajectories(
        first, second, report,
        common=(common_frames, first_common, second_common),
        output_path=args.out,
    )


if __name__ == "__main__":
    main()