import pandas as pd
import matplotlib.pyplot as plt

# 3D çizim için gerekli
from mpl_toolkits.mplot3d import Axes3D


# =========================
# AYARLAR
# =========================
csv_path = "/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1_translation.csv"

# Eğer CSV'deki translation_x/y/z değerleri
# ilk konuma göre toplam yer değiştirme ise False kalsın.
# Eğer her satır bir önceki frame'e göre delta hareket ise True yap.
USE_INCREMENTAL_MODE = False


# =========================
# CSV OKUMA
# =========================
df = pd.read_csv(csv_path)

print("Kolonlar:")
print(df.columns.tolist())

print("\nİlk 5 satır:")
print(df.head())

# Gerekli kolonlar
required_columns = [
    "translation_x",
    "translation_y",
    "translation_z",
    "frame_numbers"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"CSV içinde '{col}' kolonu bulunamadı.")


# =========================
# FRAME NUMARASINI TEMİZLE
# frame_000001 -> 1
# =========================
df["frame"] = (
    df["frame_numbers"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(int)
)

# Sayısal kolonları garantiye al
df["translation_x"] = pd.to_numeric(df["translation_x"], errors="coerce")
df["translation_y"] = pd.to_numeric(df["translation_y"], errors="coerce")
df["translation_z"] = pd.to_numeric(df["translation_z"], errors="coerce")

# Boş/hatalı satırları temizle
df = df.dropna(subset=["translation_x", "translation_y", "translation_z", "frame"])

# Frame sırasına göre sırala
df = df.sort_values("frame").reset_index(drop=True)

if len(df) == 0:
    raise ValueError("CSV okundu ama kullanılabilir veri bulunamadı.")


# =========================
# X Y Z HAREKET VERİLERİ
# =========================
if USE_INCREMENTAL_MODE:
    # Eğer değerler frame-frame arası hareket ise bunları toplayarak konum üretir
    df["x"] = df["translation_x"].cumsum()
    df["y"] = df["translation_y"].cumsum()
    df["z"] = df["translation_z"].cumsum()
else:
    # Eğer değerler zaten ilk konuma göre yer değiştirme ise direkt kullanılır
    df["x"] = df["translation_x"]
    df["y"] = df["translation_y"]
    df["z"] = df["translation_z"]


print("\nToplam frame:", len(df))
print("Başlangıç konumu:")
print(df[["frame", "x", "y", "z"]].iloc[0])

print("\nBitiş konumu:")
print(df[["frame", "x", "y", "z"]].iloc[-1])

print("\nX aralığı:", df["x"].min(), "->", df["x"].max())
print("Y aralığı:", df["y"].min(), "->", df["y"].max())
print("Z aralığı:", df["z"].min(), "->", df["z"].max())


# =========================
# 1) X-Y HAREKET HARİTASI
# =========================
plt.figure(figsize=(11, 9))

# Hareket yolu
plt.plot(
    df["x"],
    df["y"],
    linewidth=1.5,
    label="Drone Hareket Yolu"
)

# Noktalar z değerine göre renklendiriliyor
scatter = plt.scatter(
    df["x"],
    df["y"],
    c=df["z"],
    s=8,
    cmap="viridis",
    label="Frame Noktaları"
)

# Başlangıç noktası
plt.scatter(
    df["x"].iloc[0],
    df["y"].iloc[0],
    s=150,
    marker="o",
    label="Başlangıç"
)

# Bitiş noktası
plt.scatter(
    df["x"].iloc[-1],
    df["y"].iloc[-1],
    s=180,
    marker="X",
    label="Bitiş"
)

# Hareket yönünü göstermek için aralıklı oklar
arrow_step = max(len(df) // 25, 1)

x_range = df["x"].max() - df["x"].min()
y_range = df["y"].max() - df["y"].min()
head_size = max(x_range, y_range) * 0.01

for i in range(0, len(df) - arrow_step, arrow_step):
    dx = df["x"].iloc[i + arrow_step] - df["x"].iloc[i]
    dy = df["y"].iloc[i + arrow_step] - df["y"].iloc[i]

    plt.arrow(
        df["x"].iloc[i],
        df["y"].iloc[i],
        dx,
        dy,
        length_includes_head=True,
        head_width=head_size,
        alpha=0.5
    )

# Her belli aralıkta frame numarası yaz
text_step = max(len(df) // 10, 1)

for i in range(0, len(df), text_step):
    plt.text(
        df["x"].iloc[i],
        df["y"].iloc[i],
        str(df["frame"].iloc[i]),
        fontsize=8
    )

cbar = plt.colorbar(scatter)
cbar.set_label("Z Yer Değiştirmesi")

plt.xlabel("X Yer Değiştirmesi")
plt.ylabel("Y Yer Değiştirmesi")
plt.title("Drone X-Y Hareket Haritası")

plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()

plt.savefig("drone_xy_hareket_haritasi.png", dpi=300)
plt.show()


# =========================
# 2) 3D HAREKET GRAFİĞİ
# =========================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    df["x"],
    df["y"],
    df["z"],
    linewidth=1.5,
    label="3D Hareket Yolu"
)

ax.scatter(
    df["x"].iloc[0],
    df["y"].iloc[0],
    df["z"].iloc[0],
    s=120,
    marker="o",
    label="Başlangıç"
)

ax.scatter(
    df["x"].iloc[-1],
    df["y"].iloc[-1],
    df["z"].iloc[-1],
    s=150,
    marker="X",
    label="Bitiş"
)

ax.set_xlabel("X Yer Değiştirmesi")
ax.set_ylabel("Y Yer Değiştirmesi")
ax.set_zlabel("Z Yer Değiştirmesi")
ax.set_title("Drone 3D Hareket Yolu")

ax.legend()

plt.tight_layout()
plt.savefig("drone_3d_hareket_yolu.png", dpi=300)
plt.show()


# =========================
# 3) FRAME'E GÖRE X Y Z DEĞİŞİMİ
# =========================
plt.figure(figsize=(12, 8))

plt.plot(df["frame"], df["x"], label="X")
plt.plot(df["frame"], df["y"], label="Y")
plt.plot(df["frame"], df["z"], label="Z")

plt.xlabel("Frame")
plt.ylabel("Yer Değiştirme")
plt.title("Frame'e Göre X-Y-Z Yer Değişimleri")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("frame_bazli_xyz_degisim.png", dpi=300)
plt.show()


# =========================
# 4) SONUÇ ÖZETİ
# =========================
print("\nGrafikler kaydedildi:")
print("- drone_xy_hareket_haritasi.png")
print("- drone_3d_hareket_yolu.png")
print("- frame_bazli_xyz_degisim.png")