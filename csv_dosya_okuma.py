import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("/Users/mehmetyilmaz/Desktop/THYZ_2026_Ornek_Veri_1_translation.csv")

# Çıktı dosyasını oluştur
with open("/Users/mehmetyilmaz/Desktop/csv/frame_xyz_degerleri.py", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        frame_no = int(row["frame_numbers"])
        x = row["translation_x"]
        y = row["translation_y"]
        z = row["translation_z"]

        f.write(f"frame{frame_no} = [{x}, {y}, {z}]\n")