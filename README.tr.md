# WBC Analyzer

White blood cell classification için çok modlu, agentic yapay zeka sistemi.

Bu dosya projenin Türkçe dokümantasyonudur. İngilizce ana sürüm için [README.md](README.md) dosyasına bakın.

## Genel Bakış

Periferik yayma görüntülerinden beyaz kan hücresi sınıflandırması yapan uçtan uca bir derin öğrenme sistemi. Proje; özel bir DenseNet121 mimarisi, CBAM attention blokları, Grad-CAM açıklanabilirliği ve XAI özetleri üreten Gemini tabanlı raporlama akışı kullanır.

## Temel Özellikler

- WBC sınıflandırması için DenseNet121 + CBAM attention
- Ön plan maskesi ile güçlendirilmiş Grad-CAM açıklanabilirlik
- Isı haritası yorumlayan Gemini tabanlı agentic raporlama
- CLAHE, bilateral filtering ve seçici sharpen içeren medikal ön işleme
- Dengesiz medikal veri için özel loss ve aktivasyon fonksiyonları
- Sürükle-bırak inference için Flask web arayüzü

## Model Erişimi

Eğitilmiş `.keras` model dosyası boyutu nedeniyle repoya eklenmemiştir.

[Modeli indirin](https://drive.google.com/file/d/1imbnTiTTxEpuXD_HB_IJm0RxE03PJ6GQ/view?usp=sharing) ve `data/models/` klasörüne `wbc_final_model_densenet.keras` adıyla yerleştirin.

## Hızlı Başlangıç

```bash
git clone https://github.com/frissonitte/wbc-analyzer-final.git
cd wbc-final
pip install -r requirements.txt
```

`.env` dosyasına Gemini API anahtarınızı ekleyin:

```env
GEMINI_API_KEY=your_key_here
```

Uygulamayı çalıştırın:

```bash
python app.py
```

## Nihai Çıkarım

En iyi raporlanan sonucu yeniden üretmek için:

```bash
python class.py \
    --model-path data/models/wbc_final_model_densenet.keras \
    --data-root data/raabin-wbc-data \
    --output-dir outputs/final \
    --testb-binary-mode main \
    --tta light \
    --color-normalization reinhard
```

## Shortcut-Resistant Retraining

Arka plan kısa yollarını azaltmak için yeniden eğitim hattını kullanın:

```bash
python train_shortcut_resistant.py \
    --data-root data/raabin-wbc-data \
    --train-split Train \
    --val-fraction 0.15 \
    --phase1-epochs 15 \
    --phase2-epochs 15 \
    --main-loss cce \
    --label-smoothing 0.1 \
    --crop-prob 0.2 \
    --color-normalization none \
    --aux-loss-weight 1.0 \
    --model-path data/models/wbc_final_model_densenet_focus.keras
```

## Depo Notları

- Ana proje dokümantasyonu: [README.md](README.md)
- Önceki proje: [WBC-Classification-Project](https://github.com/frissonitte/WBC-Classification-Project)

## Yazar

Emirhan Yıldırım  
emirhan.yildirim2@ogr.sakarya.edu.tr  
Sakarya University, Information Systems Engineering
