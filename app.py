import os
import sys
import io
import logging
import logging.handlers
import numpy as np
import cv2
import tensorflow as tf
import traceback
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import keras
import base64

mixed_precision = keras.mixed_precision

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.custom_layers import MedSwish, WBCAttentionBlock
from src.custom_losses import WBCFocalLoss
from src.preprocessing import PreprocessingFilters

try:
    mixed_precision.set_global_policy("mixed_float16")
except:
    pass

app = Flask(__name__)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        print(f"HATA: '{last_conv_layer_name}' katmanı modelde bulunamadı!")
        return None

    # Gradient modeli
    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        print(f"DEBUG: Gradient modeli oluşturuldu. "
              f"Conv çıktı: {last_conv_layer.output.shape}, "
              f"Model çıktı: {model.output.shape}")
    except Exception as e:
        print(f"HATA: Gradient modeli oluşturulamadı: {e}")
        traceback.print_exc()
        return None

    # Mixed precision güvenliği için
    img_array = tf.cast(img_array, tf.float32)

    # Gradient hesapla
    with tf.GradientTape(watch_accessed_variables=True) as tape:
        # Conv çıktısını açıkça izle
        conv_outputs, predictions = grad_model(img_array, training=False)
        
        # float32'ye dönüştür (mixed_float16 koruma)
        conv_outputs_f32 = tf.cast(conv_outputs, tf.float32)
        predictions_f32 = tf.cast(predictions, tf.float32)

        # tape'e conv_outputs'u izlemesini söyle
        tape.watch(conv_outputs)

        if pred_index is None:
            pred_index = tf.argmax(predictions_f32[0])
        
        class_channel = predictions_f32[:, pred_index]
        
        print(f"DEBUG: pred_index={pred_index}, "
              f"class_score={class_channel.numpy()[0]:.4f}, "
              f"conv_outputs shape={conv_outputs.shape}")

    # Gradient hesapla - conv_outputs'a göre
    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        print("HATA: Gradientler hala None!")
        print("DEBUG: Alternatif yöntem deneniyor...")
        
        # ALTERNATİF YÖNTEM: Yeni tape ile tekrar dene
        try:
            with tf.GradientTape() as tape2:
                tape2.watch(img_array)
                conv_outputs2, predictions2 = grad_model(img_array, training=False)
                tape2.watch(conv_outputs2)
                predictions2 = tf.cast(predictions2, tf.float32)
                if pred_index is None:
                    pred_index = tf.argmax(predictions2[0])
                class_channel2 = predictions2[:, pred_index]
            
            grads = tape2.gradient(class_channel2, conv_outputs2)
            conv_outputs = conv_outputs2
            
            if grads is None:
                print("HATA: Alternatif yöntem de başarısız!")
                return None
            else:
                print("DEBUG: Alternatif yöntem BAŞARILI!")
        except Exception as e2:
            print(f"HATA: Alternatif yöntem exception: {e2}")
            return None

    print(f"DEBUG: Gradientler alındı! Shape: {grads.shape}, "
          f"min: {tf.reduce_min(grads).numpy():.6f}, "
          f"max: {tf.reduce_max(grads).numpy():.6f}")

    # Global Average Pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Heatmap hesabı
    conv_output_vals = conv_outputs[0]
    heatmap = conv_output_vals @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalizasyon
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    result = heatmap.numpy()
    print(f"DEBUG: Heatmap üretildi! Shape: {result.shape}, "
          f"min: {result.min():.4f}, max: {result.max():.4f}")
    return result

def get_last_conv_layer(model):
    """
    Grad-CAM için en uygun katmanı bulur.
    """
    
    known_target_layers = [
        'conv5_block16_2_conv',
        'conv5_block16_concat',
        'bn',              
        'relu',
    ]
    
    for name in known_target_layers:
        try:
            layer = model.get_layer(name)
            shape = layer.output.shape
            if len(shape) == 4:
                print(f"DEBUG: Grad-CAM katmanı (bilinen isim): '{name}', shape: {shape}")
                return name
        except (ValueError, Exception):
            continue

    skip_keywords = [
        'input', 'random_rotation', 'random_flip',
        'random_zoom', 'random_contrast', 'rescaling',
        'dropout', 'lambda', 'global', 'flatten', 'dense',
        'batch_normalization', 'softmax',
        'attention',    # ← Custom attention block'u atla
        'med_swish',    # ← Custom activation'ı atla
        'wbc_',         # ← Diğer custom layer'ları atla
    ]

    candidate = None

    for layer in model.layers:
        name_lower = layer.name.lower()

        if any(s in name_lower for s in skip_keywords):
            continue

        try:
            output_tensor = layer.output
            if hasattr(output_tensor, 'shape'):
                shape = output_tensor.shape
            else:
                continue

            if len(shape) == 4:
                h = shape[1]
                w = shape[2]
                if h is not None and w is not None and h > 1 and w > 1:
                    candidate = layer.name
        except Exception:
            continue

    if candidate:
        try:
            shape = model.get_layer(candidate).output.shape
            print(f"DEBUG: Grad-CAM katmanı (tarama): '{candidate}', shape: {shape}")
        except:
            print(f"DEBUG: Grad-CAM katmanı (tarama): '{candidate}'")
    else:
        print("DEBUG: Hiçbir uygun Grad-CAM katmanı bulunamadı!")

    return candidate

custom_objects = {
    "MedSwish": MedSwish,
    "WBCAttentionBlock": WBCAttentionBlock,
    "WBCFocalLoss": WBCFocalLoss,
    "wbc_focal_loss": WBCFocalLoss,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "wbc_final_model_densenet.keras")
print(f"Checking for model at: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print("CRITICAL: Model file really does not exist at that path!")

try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print(f"Model loaded from: {MODEL_PATH}")
    print("Model ısınma turu (Warm-up) başlatılıyor")
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    print("Model ısındı ve kullanıma hazır")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

class_names = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

class_descriptions = {
    "Basophil": "Bazofiller, inflamatuar reaksiyonlarda (özellikle alerjik) histamin salgılayan granülositlerdir. Periferik yaymada nadir görülürler (<%1).",
    "Eosinophil": "Eozinofiller, parazitik enfeksiyonlara karşı savunma ve alerjik yanıt modülasyonunda görev alır. Sitoplazmalarında turuncu-kırmızı granüller karakteristiktir.",
    "Lymphocyte": "Lenfositler (T, B, NK), adaptif bağışıklık sisteminin merkezidir. Viral enfeksiyonlarda sayıları artış gösterir. Çekirdekleri genellikle yuvarlak ve koyu boyanır.",
    "Monocyte": "Monositler, dokulara geçerek makrofajlara dönüşen ve fagositoz yapan büyük lökositlerdir. Çekirdekleri genellikle böbrek veya at nalı şeklindedir.",
    "Neutrophil": "Nötrofiller, bakteriyel enfeksiyonlara karşı ilk savunma hattını oluşturan en yaygın lökosit türüdür (%40-70). Çok loblu çekirdek yapısına sahiptirler.",
}


@app.route("/")
def home():
    return render_template("index.html")


app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp"}


@app.route("/predict", methods=["POST"])
def predict():
    app.logger.info(f"New prediction request from: {request.remote_addr}")

    if model is None:
        return jsonify({"error": "Sistem şu an bakımda (Model Yüklenemedi)."}), 500

    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı."}), 400

    file = request.files["file"]

    if file.content_type not in ALLOWED_MIME_TYPES:
        return (
            jsonify(
                {
                    "error": "Desteklenmeyen dosya formatı. Lütfen JPEG veya PNG kullanın."
                }
            ),
            415,
        )

    try:
        image_bytes = file.read()
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = np.array(pil_img)
            img = cv2.resize(img, (224, 224))

            img_processed = PreprocessingFilters.medical_enhanced(img)
            img_batch = np.expand_dims(img_processed, axis=0)

        except UnidentifiedImageError:
            return jsonify({"error": "Görüntü dosyası bozuk veya okunamıyor."}), 400

        predictions = model.predict(img_batch, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        all_probs = {
            class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
        }

        app.logger.info(f"Tahmin: {predicted_class} ({confidence:.4f})")

        # ===== GRAD-CAM BÖLÜMÜ =====
        heatmap_base64 = None
        try:
            last_conv_layer_name = get_last_conv_layer(model)
            print(f"DEBUG: Grad-CAM katmanı: {last_conv_layer_name}")

            if last_conv_layer_name:
                heatmap = make_gradcam_heatmap(
                    img_batch, model, last_conv_layer_name, predicted_index
                )
                print(f"DEBUG: Heatmap sonucu: {type(heatmap)}, "
                      f"shape: {getattr(heatmap, 'shape', 'N/A')}")

                if heatmap is not None and heatmap.size > 0:
                    heatmap = np.float32(heatmap)
                    heatmap = np.nan_to_num(heatmap)
                    
                    # Boyut düzeltme
                    if heatmap.ndim < 2:
                        print("DEBUG: Heatmap boyutu düşük, 7x7'ye reshape")
                        side = int(np.sqrt(heatmap.size))
                        if side * side == heatmap.size:
                            heatmap = heatmap.reshape(side, side)
                        else:
                            heatmap = np.zeros((7, 7), dtype=np.float32)

                    # 224x224'e resize
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                    # Orijinal görüntü ile overlay (süperimpoze)
                    original_resized = cv2.resize(np.array(pil_img), (224, 224))
                    original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
                    
                    superimposed = cv2.addWeighted(
                        original_bgr, 0.6,
                        heatmap_colored, 0.4,
                        0
                    )

                    # Base64'e encode et
                    _, buffer = cv2.imencode('.jpg', superimposed, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
                    print(f"DEBUG: Heatmap base64 uzunluğu: {len(heatmap_base64)}")
                else:
                    print("DEBUG: Heatmap None veya boş!")
            else:
                print("DEBUG: Uygun Grad-CAM katmanı bulunamadı!")
                
        except Exception as e:
            print(f"DEBUG: Grad-CAM HATA: {e}")
            traceback.print_exc()
        # ===== GRAD-CAM BÖLÜMÜ SONU =====
        
        return jsonify(
            {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "description": class_descriptions.get(predicted_class, ""),
                "all_probabilities": all_probs,
                "tech_details": {
                    "filter": "Medical Enhanced (CLAHE + Sharpening)",
                    "architecture": "DenseNet121 + Attention Block",
                },
                "heatmap": heatmap_base64
            }
        )

    except Exception as e:
        app.logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": "Analiz sırasında bir sunucu hatası oluştu."}), 500


if __name__ == "__main__":

    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_handler = logging.handlers.RotatingFileHandler(
        "api.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    log_handler.setFormatter(log_format)
    app.logger.addHandler(log_handler)
    app.logger.setLevel(logging.INFO)

    print("WBC Analiz Sistemi Başlatılıyor...")
    app.run(debug=True, host="0.0.0.0", port=5000)
