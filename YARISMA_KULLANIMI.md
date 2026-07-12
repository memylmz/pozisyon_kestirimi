# Yarışma kullanımı - Görev 2 (konum kestirimi)

`translation_estimator.py`, mevcut homografi algoritmasını video dosyası, GT CSV,
OpenCV pencereleri ve grafiklerden ayırarak kare-kare çalıştırır. Mevcut
`pozisyon_kestirimi_homography.py` değiştirilmemiştir.

Bağımlılıkları ve hızlı regresyon testlerini yarışma bilgisayarında önceden
çalıştırın:

```bash
python -m pip install -r requirements-yarisma.txt
python -m unittest -v test_yarisma.py
```

## Kare klasörüyle yerel simülasyon

```bash
python yarisma_simulasyonu.py \
  --frames "/veri/kareler" \
  --gt "/veri/translation.csv" \
  --out "/veri/tahmin.csv"
```

Sağlığın sonradan geri gelmesini test etmek için ilgili karelerin GT değerleri
CSV'de bulunmalı ve aralıklar verilmelidir:

```bash
python yarisma_simulasyonu.py \
  --frames "/veri/kareler" \
  --gt "/veri/tam_translation.csv" \
  --out "/veri/tahmin.csv" \
  --healthy-ranges "0:450,550:600"
```

## Sunucu döngüsüne ekleme

Adaptörü oturum başında **bir kez** oluşturun. Her karede yeniden oluşturmayın;
optik akışın önceki kareye ve kalibrasyon geçmişine ihtiyacı vardır.

```python
from competition_adapter import CompetitionTranslationAdapter

translation_adapter = CompetitionTranslationAdapter()

# Yarışmanın verdiği her kare için mevcut döngünüzün içinde:
health_status = frame_info["health_status"]
server_translation = (
    frame_info["translation_x"],
    frame_info["translation_y"],
    frame_info["translation_z"],
)

translation_adapter.add_to_prediction(
    prediction=prediction,
    detected_translation_cls=DetectedTranslation,
    frame_bgr=frame_bgr,  # image_url'den indirilip cv2.imdecode ile açılan BGR kare
    health_status=health_status,
    translation=server_translation,
    frame_id=frame_info["url"],
    frame_index=frame_index,  # sunucu listesindeki sıra numarası
    session_id=frame_info["session"],
    video_id=frame_info["video_name"],
)
```

Bu çağrı sağlık `1` ise sunucunun referans değerini aynen ekler. Sağlık `0` ise
homografi tabanlı kendi kestirimini ekler. Böylece her görsel için tam bir adet
`DetectedTranslation` üretilir.

`frame_id`, ağ retry'ında aynı karenin VO durumunu ikinci kez ilerletmesini
engeller. `session_id` veya `video_id` değişirse geçmiş otomatik sıfırlanır;
sıra dışı `frame_index` ise sessizce yanlış yörünge üretmek yerine hata verir.

Yeni video veya oturum başladığında geçmiş kareler karışmamalıdır:

```python
translation_adapter.reset_session()
```

## Dikkat edilmesi gerekenler

- Kareleri sunucunun verdiği sırada ve atlamadan işleyin.
- Görüntüyü OpenCV BGR `numpy.ndarray` olarak verin; yeniden boyutlandırmayın.
- Yarışmada farklı kamera parametreleri verilirse kestirimciyi
  `TranslationEstimator(camera_matrix=K, dist_coeffs=dist)` ile oluşturup
  adaptöre geçirin.
- Sağlık alanını sabit `450` sayacına çevirmeyin. Şartname süre ve kare
  sayılarının değişebileceğini söylüyor.
- Sağlık `0` iken sunucudaki translation alanlarını kalibrasyona vermeyin.
- İnternet yarışma ağında olmayacağı için `opencv-python`, `numpy` ve diğer
  bağımlılıkları `requirements-yarisma.txt` ile önceden kurun.
- Yarışma nihai SDK'sında sınıf veya metot adı değişirse sadece
  `competition_adapter.py` içindeki `add_to_prediction` bağlantısını güncelleyin;
  kestirim çekirdeğine dokunmayın.

## Şartname eşlemesi

- Başlangıç referansı: ilk sağlıklı karede gelen `(0, 0, 0)` aynen döner.
- İlk sağlıklı bölüm: referans çıktı olurken VO-metre kalibrasyonu birikir.
- Sağlıksız bölüm: yalnızca kamera görüntüsünden kestirilen değer döner.
- Sonradan sağlık tekrar `1` olursa: o karelerde referans aynen döner ve birikmiş
  VO drift'i bu konumda sıfırlanır. Sağlık yeniden `0` olduğunda kestirim son
  sağlıklı konumdan devam eder.
- Çıktı: ilk görüntüye göre metre cinsinden `translation_x/y/z`.

## Doğrulanmış yöntem sınırı

7.5 FPS'e düşürülmüş örnek veride ilk 450 kare sağlıklı, sonraki 1800 karenin
tamamı sağlıksız kabul edildiğinde yazılım 2250 kareyi çökmeden ve sonlu çıktıyla
işlemiştir. Buna rağmen yalnız görsel odometri uzun referanssız bölümde drift
biriktirir. Ara sağlıklı pencereler geldiğinde yerel yön/ölçek/Z yeniden
kalibrasyonu bunu azaltır; hiç referans dönmeyen en kötü durum için kalıcı çözüm
görüntü tabanlı loop-closure veya harita eşlemedir.
