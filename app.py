import os
import sys
import io
import logging
import logging.handlers
import socket
import numpy as np
import cv2
import tensorflow as tf
import traceback
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import keras
import base64
from google import genai
from google.genai import types
import PIL.Image
from dotenv import load_dotenv

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


def get_main_output_tensor(model):
    if not isinstance(model.output, (list, tuple)):
        return model.output

    for output_tensor in model.outputs:
        tensor_name = output_tensor.name.split(":")[0]
        if tensor_name.startswith("main_out"):
            return output_tensor

    return model.outputs[0]


def extract_main_predictions(predictions):
    if isinstance(predictions, dict):
        if "main_out" in predictions:
            main_predictions = predictions["main_out"]
        else:
            main_predictions = next(iter(predictions.values()))
    elif isinstance(predictions, (list, tuple)):
        main_predictions = predictions[0]
    else:
        main_predictions = predictions

    return np.array(main_predictions, dtype=np.float32)


def find_available_port(host, preferred_port, max_attempts=20):
    """
    Return the first bindable port starting from preferred_port.
    Falls back when Windows blocks or another process already owns the port.
    """

    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    retryable_errors = {13, 48, 98, 10013, 10048}

    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
                return port
            except OSError as exc:
                if exc.errno not in retryable_errors:
                    raise

    raise RuntimeError(
        f"{preferred_port}-{preferred_port + max_attempts - 1} araliginda uygun port bulunamadi."
    )

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        print(f"HATA: '{last_conv_layer_name}' katmani modelde bulunamadi!")
        return None

    try:
        main_output_tensor = get_main_output_tensor(model)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, main_output_tensor],
        )
    except Exception as e:
        print(f"HATA: Gradient modeli olusturulamadi: {e}")
        traceback.print_exc()
        return None

    img_array = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        predictions = tf.cast(predictions, tf.float32)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        print("HATA: Grad-CAM gradient uretilemedi.")
        return None

    conv_outputs = tf.cast(conv_outputs[0], tf.float32)
    grads = tf.cast(grads[0], tf.float32)

    # Guided Grad-CAM: negatif aktivasyon/gradient etkisini baskilar.
    guided_grads = (
        tf.cast(conv_outputs > 0, tf.float32)
        * tf.cast(grads > 0, tf.float32)
        * grads
    )
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if float(max_val.numpy()) <= 0:
        return None

    heatmap = heatmap / max_val
    return heatmap.numpy().astype(np.float32)

def generate_agent_report(predicted_class, confidence, heatmap_img_array):
    try:
        load_dotenv()
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        system_instruction = """
        Sen periferik yayma analizinde uzman bir yapay zeka hematoloğusun. 
        Sana bir kan hücresinin XAI (Açıklanabilir Yapay Zeka) Grad-CAM ısı haritası ve modelin tahmin sonuçları verilecek. 
        Resimdeki KIRMIZI ALANLAR, modelin karar verirken en çok odaklandığı hücre yapılarıdır.
        Görevlerin:
        1. Modelin tahminini ve güven skorunu belirt.
        2. Kırmızı alanların (odak noktalarının) hücrenin neresine (örn: çekirdek, sitoplazma, hücre zarı) denk geldiğini yorumla.
        3. Modelin odaklandığı yerin tıbbi olarak mantıklı olup olmadığını kısaca değerlendir.
        Raporun profesyonel, nesnel ve en fazla 3-4 cümle olmalıdır. Kesin teşhis koyma.
        """

        img_rgb = cv2.cvtColor(heatmap_img_array, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(img_rgb)
        
        prompt = f"Model bu hücrenin %{confidence * 100:.1f} ihtimalle {predicted_class} olduğunu tahmin etti. Lütfen ekteki ısı haritasını inceleyerek tıbbi bir ön rapor yaz."
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_image],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )
        )
        return response.text
    except Exception as e:
        print(f"Ajan Hatası: {e}")
        return "Yapay Zeka Raporu şu an oluşturulamadı. Lütfen model sonuçlarını manuel inceleyiniz."

def get_last_conv_layer(model):
    """
    Grad-CAM için en uygun katmanı bulur.
    """

    for layer in reversed(model.layers):
        try:
            is_conv = isinstance(layer, tf.keras.layers.Conv2D)
            if not is_conv:
                continue

            shape = layer.output.shape
            if len(shape) != 4:
                continue

            h, w = shape[1], shape[2]
            if h is not None and w is not None and h > 1 and w > 1:
                print(f"DEBUG: Grad-CAM katmani secildi: '{layer.name}', shape: {shape}")
                return layer.name
        except Exception:
            continue

    print("DEBUG: Uygun bir Conv2D katmani bulunamadi.")
    return None

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

        predictions = extract_main_predictions(model.predict(img_batch, verbose=0))
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        all_probs = {
            class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
        }

        app.logger.info(f"Tahmin: {predicted_class} ({confidence:.4f})")

        # ===== GRAD-CAM BÖLÜMÜ =====
        heatmap_base64 = None
        heatmap = None
        superimposed_img = None
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

                    # 224x224'e resize + yumusatma
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (0, 0), sigmaX=2.0)

                    # Arka plani baskilamak icin yumuşak hucre maskesi uygula.
                    original_resized = cv2.resize(np.array(pil_img), (224, 224))
                    foreground_mask = PreprocessingFilters.estimate_foreground_mask(original_resized)
                    heatmap_resized = heatmap_resized * (0.05 + 0.95 * foreground_mask)

                    max_heatmap = float(np.max(heatmap_resized))
                    if max_heatmap > 0:
                        heatmap_resized = heatmap_resized / max_heatmap

                    # Dusuk aktivasyonlari gosterme: yalnizca ust persentildeki alanlari vurgula.
                    heatmap_focus = np.power(np.clip(heatmap_resized, 0.0, 1.0), 1.6)
                    foreground_pixels = heatmap_focus[foreground_mask > 0.15]
                    if foreground_pixels.size > 50:
                        focus_threshold = float(np.percentile(foreground_pixels, 82))
                    else:
                        focus_threshold = 0.65
                    focus_threshold = float(np.clip(focus_threshold, 0.45, 0.9))

                    focus_alpha = np.clip(
                        (heatmap_focus - focus_threshold) / (1.0 - focus_threshold + 1e-6),
                        0.0,
                        1.0,
                    )
                    focus_alpha = focus_alpha * np.clip(foreground_mask, 0.0, 1.0)

                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                    # Orijinal görüntü ile overlay (süperimpoze)
                    original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)

                    mask_3ch = np.repeat(foreground_mask[:, :, np.newaxis], 3, axis=2)
                    visualization_base = (
                        original_bgr.astype(np.float32) * (0.55 + 0.45 * mask_3ch)
                    ).astype(np.uint8)

                    # Dinamik alfa ile yalnizca guclu XAI odaklarini boya.
                    alpha_3ch = np.repeat((0.78 * focus_alpha)[:, :, np.newaxis], 3, axis=2)
                    superimposed_img = (
                        visualization_base.astype(np.float32) * (1.0 - alpha_3ch)
                        + heatmap_colored.astype(np.float32) * alpha_3ch
                    ).astype(np.uint8)

                    # Base64'e encode et
                    _, buffer = cv2.imencode(
                        ".jpg", superimposed_img, [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")
                    print(f"DEBUG: Heatmap base64 uzunluğu: {len(heatmap_base64)}")
                else:
                    print("DEBUG: Heatmap None veya boş!")
            else:
                print("DEBUG: Uygun Grad-CAM katmanı bulunamadı!")
                
        except Exception as e:
            print(f"DEBUG: Grad-CAM HATA: {e}")
            traceback.print_exc()
        # ===== GRAD-CAM BÖLÜMÜ SONU =====

        agent_report = ""
        if superimposed_img is not None:
            print("Ajan rapor yazıyor...")
            agent_report = generate_agent_report(predicted_class, confidence, superimposed_img)
            print("Ajan raporu tamamlandı!")
        else:
            agent_report = "Isı haritası oluşturulamadığı için detaylı analiz yapılamadı."

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "description": class_descriptions.get(predicted_class, ""),
            "all_probabilities": all_probs,
            "tech_details": {
                "filter": "Medical Enhanced (CLAHE + Sharpening)",
                "architecture": "DenseNet121 + Attention Block",
                "xai": "Guided Grad-CAM + foreground-masked visualization",
            },
            "heatmap": heatmap_base64,
            "agent_report": agent_report
        })

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
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    debug = os.getenv("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    use_reloader = (
        os.getenv("FLASK_USE_RELOADER", "0").strip().lower() in {"1", "true", "yes", "on"}
    )

    try:
        preferred_port = int(os.getenv("FLASK_PORT", "5000"))
    except ValueError:
        preferred_port = 5000

    port = find_available_port(host, preferred_port)

    if port != preferred_port:
        print(
            f"Port {preferred_port} kullanilamiyor. Sunucu otomatik olarak {port} portuna gececek."
        )

    print(f"Sunucu http://{host}:{port} adresinde baslatiliyor")
    app.run(debug=debug, use_reloader=use_reloader, host=host, port=port)
