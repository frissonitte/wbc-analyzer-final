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
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import keras
import base64
from google import genai
from google.genai import types
from openai import OpenAI
import PIL.Image
from dotenv import load_dotenv

mixed_precision = keras.mixed_precision     

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.custom_layers import MedSwish, WBCAttentionBlock
from src.custom_losses import WBCFocalLoss
from src.preprocessing import PreprocessingFilters

try:
    # Mixed precision is optional; if unavailable, run with default precision.
    mixed_precision.set_global_policy("mixed_float16")
except:
    pass

app = Flask(__name__)
CORS(app)


def get_main_output_tensor(model):
    """Return the primary output tensor from single- or multi-output models."""
    if not isinstance(model.output, (list, tuple)):
        return model.output

    for output_tensor in model.outputs:
        tensor_name = output_tensor.name.split(":")[0]
        if tensor_name.startswith("main_out"):
            return output_tensor

    return model.outputs[0]


def extract_main_predictions(predictions):
    """Extract the main prediction array from dict/list/tensor model outputs."""
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
        f"No available port was found in the range {preferred_port}-{preferred_port + max_attempts - 1}."
    )

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Compute normalized Guided Grad-CAM heatmap for the selected class index."""

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        print(f"ERROR: layer '{last_conv_layer_name}' was not found in the model!")
        return None

    try:
        main_output_tensor = get_main_output_tensor(model)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, main_output_tensor],
        )
    except Exception as e:
        print(f"ERROR: could not build the gradient model: {e}")
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
        print("ERROR: Grad-CAM gradients could not be computed.")
        return None

    conv_outputs = tf.cast(conv_outputs[0], tf.float32)
    grads = tf.cast(grads[0], tf.float32)

    # Guided Grad-CAM suppresses negative activation and gradient effects.
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


def generate_agent_report_with_gpt4o(prompt, heatmap_img_array, system_instruction, github_token):
    """Generate report with GitHub Models gpt-4o as primary model."""
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=github_token,
    )
    ok, buffer = cv2.imencode(".jpg", heatmap_img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode heatmap image for gpt-4o request.")

    image_b64 = base64.b64encode(buffer).decode("utf-8")
    def extract_completion_text(chat_response):
        if not getattr(chat_response, "choices", None):
            return ""

        message = chat_response.choices[0].message
        content = getattr(message, "content", "") or ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for part in content:
                piece = ""
                if isinstance(part, dict):
                    piece = part.get("text") or part.get("content") or ""
                else:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str):
                        piece = part_text
                    elif part_text is not None:
                        piece = getattr(part_text, "value", "") or str(part_text)

                if piece:
                    parts.append(str(piece))
            return " ".join(parts).strip()

        return str(content).strip() if content else ""

    try:
        multimodal_response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {
                    "role": "developer",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            max_tokens=240,
        )
        text = extract_completion_text(multimodal_response)
        if text:
            return text
    except Exception as e:
        print(
            "gpt-4o multimodal call failed (image may be unsupported). "
            f"Falling back to text-only: {e}"
        )

    print("gpt-4o multimodal response was empty. Retrying with text-only fallback.")
    text_only_prompt = (
        f"{prompt} "
        "If image processing is unavailable, provide a cautious preliminary comment "
        "using only the prediction information and clearly recommend manual review."
    )
    text_only_response = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {
                "role": "developer",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": text_only_prompt,
            },
        ],
        max_tokens=240,
    )
    text = extract_completion_text(text_only_response)
    if text:
        return text

    raise RuntimeError("gpt-4o returned empty content for both multimodal and text-only attempts.")


def get_env_token(*names):
    """Return the first non-empty token value among environment variable names."""
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue

        normalized = value.strip().strip('"').strip("'")
        if normalized:
            return normalized

    return ""


# Class-specific morphology anchors to reduce generic report patterns.
CLASS_MORPHOLOGY_CONTEXT = {
    "Neutrophil": (
        "Neutrophils are identified by their multi-lobed nucleus (typically 3-5 lobes) "
        "and pale pink cytoplasm with fine granules. The model should focus on the nuclear "
        "lobes and their interconnecting filaments, not on background erythrocytes."
    ),
    "Lymphocyte": (
        "Lymphocytes have a large, round, darkly stained nucleus that nearly fills the cell, "
        "with only a thin rim of pale blue cytoplasm. Correct model focus should be on the "
        "nucleus-to-cytoplasm ratio and nuclear chromatin density."
    ),
    "Monocyte": (
        "Monocytes are the largest leukocytes with a kidney-shaped or horseshoe-shaped nucleus "
        "and abundant gray-blue cytoplasm that may contain vacuoles. The nucleus shape and "
        "cytoplasmic texture are the key discriminating features."
    ),
    "Eosinophil": (
        "Eosinophils have a bilobed nucleus and distinctive large, bright orange-red granules "
        "filling the cytoplasm. The granule pattern and bilobed nucleus, not background cells, "
        "should be the model focus."
    ),
    "Basophil": (
        "Basophils are rare and characterized by large, dark purple-blue granules that may "
        "obscure the nucleus. Correct focus should be on these coarse, deeply stained granules "
        "and the irregular nuclear outline beneath them."
    ),
}


SYSTEM_INSTRUCTION = """You are a clinical AI assistant supporting hematology laboratory review.
Your role is to interpret Grad-CAM explainability heatmaps produced by a white blood cell classification model.

In the provided overlay image:
- RED/YELLOW areas = regions the model weighted most heavily when making its decision
- BLUE/DARK areas = regions the model largely ignored

Your response must follow this structure, but write it as flowing prose, not as a numbered list:
1. One sentence stating the predicted class and confidence level.
2. One to two sentences describing which morphological structures the heatmap highlights (nucleus, cytoplasm, granules, cell membrane, background, etc.) and whether this aligns with the expected discriminating features for this cell type.
3. One sentence giving a clinical plausibility verdict: either confirm the focus is medically sound, flag a concern if the model appears to focus on background erythrocytes or staining artifacts, or note uncertainty if the heatmap is diffuse.

Constraints:
- Do not provide a definitive diagnosis.
- Do not repeat the class name more than twice.
- Avoid generic phrases like 'the model performed well' unless genuinely warranted.
- Keep the total response under 80 words.
- Write in the same language as the prompt you receive.
"""


def build_agent_prompt(predicted_class, confidence):
    """Build a class-aware prompt so each report uses distinct morphology anchors."""
    context = CLASS_MORPHOLOGY_CONTEXT.get(predicted_class, "")
    if confidence >= 0.90:
        confidence_descriptor = "high"
    elif confidence >= 0.70:
        confidence_descriptor = "moderate"
    else:
        confidence_descriptor = "low"

    return (
        f"The classification model predicted '{predicted_class}' with {confidence_descriptor} "
        f"confidence ({confidence * 100:.1f}%). "
        f"Morphological reference for this cell type: {context} "
        "Based on the Grad-CAM overlay, assess whether the model spatial attention "
        "aligns with these expected morphological features or shows signs of background dependency."
    )

def generate_agent_report(predicted_class, confidence, heatmap_img_array):
    """Generate a concise LLM-based interpretation report from XAI overlay image."""
    try:
        load_dotenv()
        gemini_api_key = get_env_token("GEMINI_API_KEY")
        # Support both names to avoid breaking existing local setups.
        github_token = get_env_token("GITHUB_TOKEN")
        if not gemini_api_key and not github_token:
            return {
                "text": (
                    "The AI report is unavailable because no LLM token is configured "
                    "(expected GEMINI_API_KEY or GITHUB_TOKEN)."
                ),
                "provider": "No LLM token",
            }

        img_rgb = cv2.cvtColor(heatmap_img_array, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(img_rgb)
        
        prompt = build_agent_prompt(predicted_class, confidence)

        primary_error = None
        gemini_error = None

        if github_token:
            try:
                print("Trying GitHub Models primary (openai/gpt-4o)...")
                response_text = generate_agent_report_with_gpt4o(
                    prompt,
                    heatmap_img_array,
                    SYSTEM_INSTRUCTION,
                    github_token,
                )
                if response_text:
                    return {
                        "text": response_text,
                        "provider": "OpenAI GPT-4o",
                    }
                print("gpt-4o primary returned empty content.")
            except Exception as e:
                primary_error = e
                print(f"gpt-4o primary error: {e}")

        if gemini_api_key:
            try:
                print("Trying Gemini fallback (gemini-2.5-flash)...")
                gemini_client = genai.Client(api_key=gemini_api_key)
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt, pil_image],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                    ),
                )
                if response.text:
                    return {
                        "text": response.text,
                        "provider": "Gemini 2.5 Flash",
                    }
                print("Gemini fallback returned empty content.")
            except Exception as e:
                gemini_error = e
                print(f"Gemini fallback error: {e}")

        if gemini_error is not None:
            print(f"Agent error: {gemini_error}")
        elif primary_error is not None:
            print(f"Agent error: {primary_error}")

        return {
            "text": "The AI report could not be generated right now. Please review the model output manually.",
            "provider": "Rule-based fallback",
        }
    except Exception as e:
        print(f"Agent error: {e}")
        return {
            "text": "The AI report could not be generated right now. Please review the model output manually.",
            "provider": "Rule-based fallback",
        }

def get_last_conv_layer(model):
    """
    Find the most suitable layer for Grad-CAM.
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
                print(f"DEBUG: selected Grad-CAM layer: '{layer.name}', shape: {shape}")
                return layer.name
        except Exception:
            continue

    print("DEBUG: no suitable Conv2D layer was found.")
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
    # Warm-up avoids first-request latency spikes in production-like usage.
    print("Starting model warm-up pass.")
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    _ = model.predict(dummy_input, verbose=0)
    print("Model warmed up and ready to use.")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

class_names = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

class_descriptions = {
    "Basophil": "Basophils are granulocytes that release histamine in inflammatory reactions (especially allergic). They are rarely seen in peripheral blood smears (<1%).",
    "Eosinophil": "Eosinophils are involved in defense against parasitic infections and modulation of allergic responses. Orange-red granules are characteristic of their cytoplasm.",
    "Lymphocyte": "Lymphocytes (T, B, NK) are central to the adaptive immune system. Their numbers increase in viral infections. Their nuclei are usually round and darkly stained.",
    "Monocyte": "Monocytes are large leukocytes that migrate to tissues, transform into macrophages, and perform phagocytosis. Their nuclei are usually kidney-shaped or horseshoe-shaped.",
    "Neutrophil": "Neutrophils are the most common type of leukocyte (40-70%), forming the first line of defense against bacterial infections. They have a multilobed nucleus.",
}


@app.route("/")
def home():
    """Serve the single-page web UI."""
    return render_template("index.html")


app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp"}


@app.route("/predict", methods=["POST"])
def predict():
    """Run preprocessing, model inference, Grad-CAM generation, and report synthesis."""
    app.logger.info(f"New prediction request from: {request.remote_addr}")

    if model is None:
        return jsonify({"error": "The system is currently under maintenance (model could not be loaded)."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file was found."}), 400

    file = request.files["file"]

    if file.content_type not in ALLOWED_MIME_TYPES:
        return (
            jsonify(
                {
                    "error": "Unsupported file format. Please use JPEG or PNG."
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
            return jsonify({"error": "The image file is corrupted or unreadable."}), 400

        predictions = extract_main_predictions(model.predict(img_batch, verbose=0))
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index])

        # Keep all class probabilities for chart rendering on the client.
        all_probs = {
            class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
        }

        app.logger.info(f"Prediction: {predicted_class} ({confidence:.4f})")

        # ===== GRAD-CAM SECTION =====
        heatmap_base64 = None
        heatmap = None
        superimposed_img = None
        try:
            last_conv_layer_name = get_last_conv_layer(model)
            print(f"DEBUG: Grad-CAM layer: {last_conv_layer_name}")

            if last_conv_layer_name:
                heatmap = make_gradcam_heatmap(
                    img_batch, model, last_conv_layer_name, predicted_index
                )
                print(f"DEBUG: heatmap result: {type(heatmap)}, "
                      f"shape: {getattr(heatmap, 'shape', 'N/A')}")

                if heatmap is not None and heatmap.size > 0:
                    heatmap = np.float32(heatmap)
                    heatmap = np.nan_to_num(heatmap)
                    
                    # Fix heatmap dimensions when needed.
                    if heatmap.ndim < 2:
                        print("DEBUG: heatmap dimension is too small; reshaping to 7x7.")
                        side = int(np.sqrt(heatmap.size))
                        if side * side == heatmap.size:
                            heatmap = heatmap.reshape(side, side)
                        else:
                            heatmap = np.zeros((7, 7), dtype=np.float32)

                    # Resize to 224x224 and smooth the heatmap.
                    heatmap_resized = cv2.resize(heatmap, (224, 224))
                    heatmap_resized = cv2.GaussianBlur(heatmap_resized, (0, 0), sigmaX=2.0)

                    # Apply a soft cell mask to suppress background influence.
                    original_resized = cv2.resize(np.array(pil_img), (224, 224))
                    foreground_mask = PreprocessingFilters.estimate_foreground_mask(original_resized)
                    heatmap_resized = heatmap_resized * (0.05 + 0.95 * foreground_mask)

                    max_heatmap = float(np.max(heatmap_resized))
                    if max_heatmap > 0:
                        heatmap_resized = heatmap_resized / max_heatmap

                    # Hide low activations and emphasize only the upper percentile regions.
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

                    # Overlay the heatmap on top of the original image.
                    original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)

                    mask_3ch = np.repeat(foreground_mask[:, :, np.newaxis], 3, axis=2)
                    visualization_base = (
                        original_bgr.astype(np.float32) * (0.55 + 0.45 * mask_3ch)
                    ).astype(np.uint8)

                    # Use dynamic alpha to paint only strong XAI focus regions.
                    alpha_3ch = np.repeat((0.78 * focus_alpha)[:, :, np.newaxis], 3, axis=2)
                    superimposed_img = (
                        visualization_base.astype(np.float32) * (1.0 - alpha_3ch)
                        + heatmap_colored.astype(np.float32) * alpha_3ch
                    ).astype(np.uint8)

                    # Encode the visualization as Base64.
                    _, buffer = cv2.imencode(
                        ".jpg", superimposed_img, [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")
                    print(f"DEBUG: heatmap Base64 length: {len(heatmap_base64)}")
                else:
                    print("DEBUG: heatmap is None or empty!")
            else:
                print("DEBUG: no suitable Grad-CAM layer was found!")
                
        except Exception as e:
            print(f"DEBUG: Grad-CAM error: {e}")
            traceback.print_exc()
        # ===== END GRAD-CAM SECTION =====

        agent_report = ""
        agent_report_model = "Unknown"
        if superimposed_img is not None:
            print("Generating agent report...")
            agent_result = generate_agent_report(predicted_class, confidence, superimposed_img)
            if isinstance(agent_result, dict):
                agent_report = str(agent_result.get("text", "")).strip()
                agent_report_model = str(agent_result.get("provider", "Unknown")).strip()
            else:
                agent_report = str(agent_result).strip()
                agent_report_model = "Unknown"
            print("Agent report completed.")
        else:
            agent_report = "Detailed analysis could not be performed because the heatmap could not be generated."
            agent_report_model = "N/A (Heatmap unavailable)"

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
            "agent_report": agent_report,
            "agent_report_model": agent_report_model,
        })

    except Exception as e:
        app.logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": "A server error occurred during analysis."}), 500


if __name__ == "__main__":

    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    log_handler = logging.handlers.RotatingFileHandler(
        "api.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    log_handler.setFormatter(log_format)
    app.logger.addHandler(log_handler)
    app.logger.setLevel(logging.INFO)

    print("Starting WBC analysis system...")
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
            f"Port {preferred_port} is unavailable. The server will automatically switch to port {port}."
        )

    print(f"Server starting at http://{host}:{port}")
    app.run(debug=debug, use_reloader=use_reloader, host=host, port=port)
