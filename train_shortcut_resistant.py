import argparse
import math
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

keras = tf.keras
layers = keras.layers

from src.custom_layers import MedSwish, WBCAttentionBlock
from src.custom_losses import WBCFocalLoss
from src.preprocessing import PreprocessingFilters

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAIN_OUTPUT_NAME = "main_out"
AUX_OUTPUT_NAME = "aux_binary_out"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WBC model with shortcut-resistant pipeline (foreground crop + bg randomization + XAI focus monitor)."
    )
    parser.add_argument("--data-root", default="data/raabin-wbc-data", help="Dataset root directory.")
    parser.add_argument("--train-split", default="Train", help="Training split folder under data root.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction sampled from Train split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--phase1-epochs", type=int, default=15, help="Feature extraction epochs.")
    parser.add_argument("--phase2-epochs", type=int, default=15, help="Fine-tuning epochs.")
    parser.add_argument("--model-path", default="data/models/wbc_final_model_densenet.keras", help="Output model path.")
    parser.add_argument("--results-dir", default="results_shortcut_resistant", help="Directory for plots and logs.")
    parser.add_argument("--crop-prob", type=float, default=0.2, help="Probability of applying foreground crop during training.")
    parser.add_argument("--bg-randomization-prob", type=float, default=0.15, help="Probability of applying background randomization during training.")
    parser.add_argument("--bg-randomization-strength", type=float, default=0.35, help="Mixing strength for synthetic background on non-foreground regions.")
    parser.add_argument("--stain-jitter-prob", type=float, default=0.3, help="Probability of mild color/stain jitter during training.")
    parser.add_argument("--neutrophil-aug-scale", type=float, default=0.3, help="Scale factor for Neutrophil crop/bg augmentation probabilities.")
    parser.add_argument("--lymphocyte-aug-scale", type=float, default=0.6, help="Scale factor for Lymphocyte crop/bg augmentation probabilities.")
    parser.add_argument("--disable-aux-binary-head", action="store_true", help="Disable auxiliary binary head and train as single-head model.")
    parser.add_argument("--aux-loss-weight", type=float, default=1.0, help="Loss weight for auxiliary binary head.")
    parser.add_argument("--aux-positive-class", default="Neutrophil", help="Positive class name for auxiliary binary head.")
    parser.add_argument("--aux-negative-class", default="Lymphocyte", help="Negative class name for auxiliary binary head.")
    parser.add_argument("--color-normalization", choices=["none", "reinhard"], default="none", help="Color normalization strategy to reduce camera/stain shift.")
    parser.add_argument("--normalization-reference-samples", type=int, default=1200, help="Number of train images used to estimate normalization reference statistics.")
    parser.add_argument("--main-loss", choices=["focal", "cce"], default="cce", help="Main head loss function.")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for CCE main loss.")
    parser.add_argument("--focal-class-weights", default="1.2,1.1,1.0,1.1,1.15", help="Comma-separated focal loss class weights in class order.")
    parser.add_argument("--xai-sample-count", type=int, default=24, help="Validation sample count used for XAI focus monitoring.")
    parser.add_argument("--xai-every-n-epochs", type=int, default=2, help="Compute XAI focus ratio every N epochs to reduce training overhead.")
    parser.add_argument("--xai-focus-threshold", type=float, default=0.55, help="Minimum acceptable mean XAI foreground focus ratio.")
    parser.add_argument("--xai-patience", type=int, default=3, help="Stop if XAI focus ratio stays below threshold for this many epochs.")
    parser.add_argument("--dry-run", action="store_true", help="Run one batch sanity-check and exit.")
    return parser.parse_args()


def parse_class_weights(raw_value, expected_count):
    try:
        weights = [float(x.strip()) for x in raw_value.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError("Invalid focal class weights format.") from exc

    if len(weights) != expected_count:
        raise ValueError(
            f"Expected {expected_count} class weights but received {len(weights)}."
        )

    return weights


def build_main_head_loss(args, class_weights):
    if args.main_loss == "focal":
        return WBCFocalLoss(class_weights=class_weights)

    smoothing = float(np.clip(args.label_smoothing, 0.0, 0.3))
    return keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)


def get_main_output_tensor(model):
    if not isinstance(model.output, (list, tuple)):
        return model.output

    for output_tensor in model.outputs:
        tensor_name = output_tensor.name.split(":")[0]
        if tensor_name.startswith(MAIN_OUTPUT_NAME):
            return output_tensor

    return model.outputs[0]


def extract_main_predictions(predictions):
    if isinstance(predictions, dict):
        if MAIN_OUTPUT_NAME in predictions:
            main_predictions = predictions[MAIN_OUTPUT_NAME]
        else:
            main_predictions = next(iter(predictions.values()))
    elif isinstance(predictions, (list, tuple)):
        main_predictions = predictions[0]
    else:
        main_predictions = predictions

    return np.array(main_predictions, dtype=np.float32)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_class_names(train_dir):
    names = [p.name for p in Path(train_dir).iterdir() if p.is_dir()]
    if not names:
        raise ValueError(f"No class directories found in: {train_dir}")
    return sorted(names)


def collect_samples(train_dir, class_names):
    samples = []
    for class_index, class_name in enumerate(class_names):
        class_dir = Path(train_dir) / class_name
        if not class_dir.exists():
            continue
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(file_path), class_index))
    if not samples:
        raise ValueError(f"No images found in: {train_dir}")
    return samples


def split_train_val(samples, num_classes, val_fraction, seed):
    rng = np.random.default_rng(seed)
    samples_by_class = [[] for _ in range(num_classes)]
    for path, label in samples:
        samples_by_class[label].append((path, label))

    train_samples = []
    val_samples = []

    for class_items in samples_by_class:
        rng.shuffle(class_items)
        n_val = max(1, int(len(class_items) * val_fraction)) if len(class_items) > 1 else 0
        val_items = class_items[:n_val]
        train_items = class_items[n_val:]

        if not train_items and val_items:
            train_items = [val_items.pop()]

        train_samples.extend(train_items)
        val_samples.extend(val_items)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


def load_rgb_image(image_path, image_size):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    return image


def estimate_reinhard_reference(samples, image_size, seed, max_samples):
    rng = np.random.default_rng(seed)

    all_paths = [path for path, _ in samples]
    if not all_paths:
        return None, None

    n = min(max_samples, len(all_paths))
    if n <= 0:
        return None, None

    selected_paths = list(rng.choice(all_paths, size=n, replace=False))

    means = []
    stds = []

    for image_path in selected_paths:
        image = load_rgb_image(image_path, image_size)
        if image is None:
            continue

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        means.append(np.mean(lab, axis=(0, 1)))
        stds.append(np.std(lab, axis=(0, 1)) + 1e-6)

    if not means:
        return None, None

    target_mean = np.mean(np.array(means, dtype=np.float32), axis=0)
    target_std = np.mean(np.array(stds, dtype=np.float32), axis=0)

    return target_mean.astype(np.float32), target_std.astype(np.float32)


def apply_reinhard_normalization(image, target_mean, target_std):
    if target_mean is None or target_std is None:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean = np.mean(lab, axis=(0, 1))
    src_std = np.std(lab, axis=(0, 1)) + 1e-6

    normalized = (lab - src_mean) / src_std
    normalized = normalized * target_std + target_mean
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)


def crop_to_foreground(image, mask, rng):
    binary = mask > 0.2
    if not np.any(binary):
        return image

    ys, xs = np.where(binary)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    h, w = image.shape[:2]
    box_h = y_max - y_min + 1
    box_w = x_max - x_min + 1

    if box_h < int(0.15 * h) or box_w < int(0.15 * w):
        return image

    margin = int(0.22 * max(box_h, box_w))
    jitter = int(0.03 * max(box_h, box_w))

    y_min = max(0, y_min - margin - int(rng.integers(-jitter, jitter + 1)))
    x_min = max(0, x_min - margin - int(rng.integers(-jitter, jitter + 1)))
    y_max = min(h - 1, y_max + margin + int(rng.integers(-jitter, jitter + 1)))
    x_max = min(w - 1, x_max + margin + int(rng.integers(-jitter, jitter + 1)))

    crop = image[y_min : y_max + 1, x_min : x_max + 1]
    if crop.size == 0:
        return image

    return cv2.resize(crop, (w, h))


def apply_stain_jitter(image, rng, prob):
    if float(rng.random()) > prob:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

    l_shift = float(rng.uniform(-8.0, 8.0))
    a_shift = float(rng.uniform(-6.0, 6.0))
    b_shift = float(rng.uniform(-6.0, 6.0))

    lab[:, :, 0] = np.clip(lab[:, :, 0] + l_shift, 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def randomize_background(image, mask, rng, prob, strength):
    if float(rng.random()) > prob:
        return image

    strength = float(np.clip(strength, 0.0, 0.85))
    if strength <= 0:
        return image

    h, w = image.shape[:2]
    sigma = float(rng.uniform(2.5, 6.5))
    background = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)

    bg_color = rng.integers(175, 235, size=(1, 1, 3), dtype=np.uint8)
    tint_mix = float(rng.uniform(0.08, 0.22))
    background = np.clip(
        background.astype(np.float32) * (1.0 - tint_mix)
        + bg_color.astype(np.float32) * tint_mix,
        0,
        255,
    ).astype(np.uint8)

    bg_noise = rng.normal(0.0, 7.0, size=(h, w, 3)).astype(np.float32)
    background = np.clip(background.astype(np.float32) + bg_noise, 0, 255).astype(np.uint8)

    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    outside_alpha = (1.0 - mask_3ch) * strength
    mixed = (
        image.astype(np.float32) * (1.0 - outside_alpha)
        + background.astype(np.float32) * outside_alpha
    )
    return np.clip(mixed, 0, 255).astype(np.uint8)


def preprocess_for_training(
    image,
    rng,
    label_name,
    color_normalization,
    target_lab_mean,
    target_lab_std,
    crop_prob,
    bg_randomization_prob,
    bg_randomization_strength,
    stain_jitter_prob,
    neutrophil_aug_scale,
    lymphocyte_aug_scale,
):
    if color_normalization == "reinhard":
        image = apply_reinhard_normalization(image, target_lab_mean, target_lab_std)

    class_crop_prob = crop_prob
    class_bg_prob = bg_randomization_prob

    if label_name == "Neutrophil":
        class_crop_prob *= float(np.clip(neutrophil_aug_scale, 0.0, 1.0))
        class_bg_prob *= float(np.clip(neutrophil_aug_scale, 0.0, 1.0))
    elif label_name == "Lymphocyte":
        class_crop_prob *= float(np.clip(lymphocyte_aug_scale, 0.0, 1.0))
        class_bg_prob *= float(np.clip(lymphocyte_aug_scale, 0.0, 1.0))

    mask = PreprocessingFilters.estimate_foreground_mask(image)

    if float(rng.random()) < class_crop_prob:
        image = crop_to_foreground(image, mask, rng)
        mask = PreprocessingFilters.estimate_foreground_mask(image)

    image = apply_stain_jitter(image, rng, prob=stain_jitter_prob)

    image = randomize_background(
        image,
        mask,
        rng,
        prob=class_bg_prob,
        strength=bg_randomization_strength,
    )

    return PreprocessingFilters.medical_enhanced(image)


def preprocess_for_eval(image, color_normalization, target_lab_mean, target_lab_std):
    if color_normalization == "reinhard":
        image = apply_reinhard_normalization(image, target_lab_mean, target_lab_std)

    return PreprocessingFilters.medical_enhanced(image)


class WBCSequence(keras.utils.Sequence):
    def __init__(
        self,
        samples,
        class_names,
        batch_size,
        image_size,
        seed,
        training,
        enable_aux_binary_head,
        aux_positive_index,
        aux_negative_index,
        color_normalization,
        target_lab_mean,
        target_lab_std,
        crop_prob,
        bg_randomization_prob,
        bg_randomization_strength,
        stain_jitter_prob,
        neutrophil_aug_scale,
        lymphocyte_aug_scale,
    ):
        super().__init__()
        self.samples = list(samples)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.batch_size = batch_size
        self.image_size = image_size
        self.training = training
        self.enable_aux_binary_head = enable_aux_binary_head
        self.aux_positive_index = aux_positive_index
        self.aux_negative_index = aux_negative_index
        self.color_normalization = color_normalization
        self.target_lab_mean = target_lab_mean
        self.target_lab_std = target_lab_std
        self.crop_prob = crop_prob
        self.bg_randomization_prob = bg_randomization_prob
        self.bg_randomization_strength = bg_randomization_strength
        self.stain_jitter_prob = stain_jitter_prob
        self.neutrophil_aug_scale = neutrophil_aug_scale
        self.lymphocyte_aug_scale = lymphocyte_aug_scale
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.samples))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def on_epoch_end(self):
        if self.training:
            self.rng.shuffle(self.indices)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]

        images = []
        labels = []

        for sample_index in batch_indices:
            image_path, label = self.samples[sample_index]
            image = load_rgb_image(image_path, self.image_size)
            if image is None:
                continue

            if self.training:
                processed = preprocess_for_training(
                    image,
                    self.rng,
                    label_name=self.class_names[label],
                    color_normalization=self.color_normalization,
                    target_lab_mean=self.target_lab_mean,
                    target_lab_std=self.target_lab_std,
                    crop_prob=self.crop_prob,
                    bg_randomization_prob=self.bg_randomization_prob,
                    bg_randomization_strength=self.bg_randomization_strength,
                    stain_jitter_prob=self.stain_jitter_prob,
                    neutrophil_aug_scale=self.neutrophil_aug_scale,
                    lymphocyte_aug_scale=self.lymphocyte_aug_scale,
                )
            else:
                processed = preprocess_for_eval(
                    image,
                    color_normalization=self.color_normalization,
                    target_lab_mean=self.target_lab_mean,
                    target_lab_std=self.target_lab_std,
                )

            images.append(processed.astype(np.float32))
            labels.append(label)

        if not images:
            images = [np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)]
            labels = [0]

        x = np.array(images, dtype=np.float32)
        y_main = keras.utils.to_categorical(labels, num_classes=self.num_classes)

        if not self.enable_aux_binary_head:
            return x, y_main

        y_aux = np.zeros((len(labels), 1), dtype=np.float32)
        aux_sample_weight = np.zeros((len(labels),), dtype=np.float32)
        main_sample_weight = np.ones((len(labels),), dtype=np.float32)

        for idx, class_index in enumerate(labels):
            if class_index == self.aux_positive_index:
                y_aux[idx, 0] = 1.0
                aux_sample_weight[idx] = 1.0
            elif class_index == self.aux_negative_index:
                y_aux[idx, 0] = 0.0
                aux_sample_weight[idx] = 1.0

        y_tuple = (y_main, y_aux)
        sample_weight_tuple = (main_sample_weight, aux_sample_weight)
        return x, y_tuple, sample_weight_tuple


def build_model(num_classes, image_size, enable_aux_binary_head):
    inputs = layers.Input(shape=(image_size, image_size, 3))

    x = layers.RandomRotation(0.2)(inputs)
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    base_model = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_tensor=x,
    )
    base_model.trainable = False

    x = base_model.output
    x = WBCAttentionBlock(name="attention_block")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = MedSwish()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = MedSwish()(x)
    x = layers.Dropout(0.3)(x)

    output_main = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name=MAIN_OUTPUT_NAME,
    )(x)

    if enable_aux_binary_head:
        output_aux = layers.Dense(
            1,
            activation="sigmoid",
            dtype="float32",
            name=AUX_OUTPUT_NAME,
        )(x)
        outputs = [output_main, output_aux]
    else:
        outputs = output_main

    model = keras.Model(inputs=inputs, outputs=outputs, name="WBC_DenseNet121_ShortcutResistant")
    return model, base_model


def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            shape = layer.output.shape
            if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                if shape[1] > 1 and shape[2] > 1:
                    return layer.name
    return None


def make_gradcam_heatmap(img_batch, model, last_conv_layer_name, pred_index):
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        return None

    main_output_tensor = get_main_output_tensor(model)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, main_output_tensor],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return None

    conv_outputs = tf.cast(conv_outputs[0], tf.float32)
    grads = tf.cast(grads[0], tf.float32)

    guided_grads = (
        tf.cast(conv_outputs > 0, tf.float32)
        * tf.cast(grads > 0, tf.float32)
        * grads
    )
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = float(tf.reduce_max(heatmap).numpy())
    if max_val <= 0:
        return None

    return (heatmap / max_val).numpy().astype(np.float32)


def compute_xai_focus_ratio(
    model,
    sample_paths,
    image_size,
    last_conv_layer_name,
    color_normalization,
    target_lab_mean,
    target_lab_std,
):
    ratios = []

    for image_path in sample_paths:
        image = load_rgb_image(image_path, image_size)
        if image is None:
            continue

        model_input = preprocess_for_eval(
            image,
            color_normalization=color_normalization,
            target_lab_mean=target_lab_mean,
            target_lab_std=target_lab_std,
        )
        model_input = np.expand_dims(model_input, axis=0).astype(np.float32)

        predictions = extract_main_predictions(model.predict(model_input, verbose=0))
        pred_index = int(np.argmax(predictions[0]))

        heatmap = make_gradcam_heatmap(model_input, model, last_conv_layer_name, pred_index)
        if heatmap is None:
            continue

        heatmap = cv2.resize(heatmap, (image_size, image_size))
        heatmap = np.clip(heatmap, 0.0, 1.0)
        mask = PreprocessingFilters.estimate_foreground_mask(image)

        numerator = float(np.sum(heatmap * mask))
        denominator = float(np.sum(heatmap) + 1e-8)
        ratios.append(numerator / denominator)

    if not ratios:
        return None, None, 0

    return float(np.mean(ratios)), float(np.std(ratios)), len(ratios)


class XAIFocusMonitor(keras.callbacks.Callback):
    def __init__(
        self,
        sample_paths,
        image_size,
        threshold,
        patience,
        every_n_epochs,
        color_normalization,
        target_lab_mean,
        target_lab_std,
    ):
        super().__init__()
        self.sample_paths = sample_paths
        self.image_size = image_size
        self.threshold = threshold
        self.patience = patience
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.color_normalization = color_normalization
        self.target_lab_mean = target_lab_mean
        self.target_lab_std = target_lab_std
        self.low_focus_streak = 0
        self.last_conv_layer_name = None

    def on_train_begin(self, logs=None):
        self.last_conv_layer_name = get_last_conv_layer_name(self.model)
        if self.last_conv_layer_name is None:
            print("[XAI] Warning: no valid Conv2D layer found. Focus monitoring disabled.")

    def on_epoch_end(self, epoch, logs=None):
        if self.last_conv_layer_name is None:
            return

        if (epoch + 1) % self.every_n_epochs != 0:
            return

        mean_focus, std_focus, count = compute_xai_focus_ratio(
            self.model,
            self.sample_paths,
            self.image_size,
            self.last_conv_layer_name,
            self.color_normalization,
            self.target_lab_mean,
            self.target_lab_std,
        )

        if mean_focus is None:
            print("[XAI] Warning: could not compute focus ratio for this epoch.")
            return

        if logs is not None:
            logs["xai_focus_ratio"] = mean_focus

        print(
            f"[XAI] epoch={epoch + 1} focus_ratio={mean_focus:.4f} std={std_focus:.4f} n={count}"
        )

        if mean_focus < self.threshold:
            self.low_focus_streak += 1
            print(
                f"[XAI] Warning: focus ratio below threshold {self.threshold:.2f} "
                f"({self.low_focus_streak}/{self.patience})."
            )
        else:
            self.low_focus_streak = 0

        if self.low_focus_streak >= self.patience:
            print("[XAI] Early stop triggered due to persistent background-focused attention.")
            self.model.stop_training = True


def plot_history(history_phase1, history_phase2, output_path):
    phase1_acc = history_phase1.history.get("main_out_accuracy", history_phase1.history.get("accuracy", []))
    phase2_acc = history_phase2.history.get("main_out_accuracy", history_phase2.history.get("accuracy", []))
    acc = phase1_acc + phase2_acc
    val_acc = history_phase1.history.get("val_main_out_accuracy", history_phase1.history.get("val_accuracy", [])) + history_phase2.history.get("val_main_out_accuracy", history_phase2.history.get("val_accuracy", []))
    loss = history_phase1.history.get("loss", []) + history_phase2.history.get("loss", [])
    val_loss = history_phase1.history.get("val_loss", []) + history_phase2.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    phase_boundary = len(phase1_acc) - 1
    if phase_boundary >= 0:
        plt.axvline(phase_boundary, linestyle="--", label="fine_tune_start")
    plt.title("Accuracy")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    if phase_boundary >= 0:
        plt.axvline(phase_boundary, linestyle="--", label="fine_tune_start")
    plt.title("Loss")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_split_summary(summary_path, class_names, train_samples, val_samples):
    train_counts = {name: 0 for name in class_names}
    val_counts = {name: 0 for name in class_names}

    for _, label in train_samples:
        train_counts[class_names[label]] += 1
    for _, label in val_samples:
        val_counts[class_names[label]] += 1

    lines = ["class,train_count,val_count"]
    for name in class_names:
        lines.append(f"{name},{train_counts[name]},{val_counts[name]}")

    Path(summary_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    set_global_seed(args.seed)

    train_dir = Path(args.data_root) / args.train_split
    if not train_dir.exists():
        raise FileNotFoundError(f"Train split directory does not exist: {train_dir}")

    class_names = list_class_names(train_dir)
    all_samples = collect_samples(train_dir, class_names)
    train_samples, val_samples = split_train_val(
        all_samples,
        num_classes=len(class_names),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"Classes: {class_names}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples (from Train): {len(val_samples)}")

    enable_aux_binary_head = not args.disable_aux_binary_head

    aux_positive_index = None
    aux_negative_index = None
    if enable_aux_binary_head:
        if args.aux_positive_class == args.aux_negative_class:
            print("Auxiliary head disabled: positive and negative class names must be different.")
            enable_aux_binary_head = False
        elif args.aux_positive_class not in class_names or args.aux_negative_class not in class_names:
            print(
                "Auxiliary head disabled: requested classes not found in dataset."
            )
            enable_aux_binary_head = False
        else:
            aux_positive_index = class_names.index(args.aux_positive_class)
            aux_negative_index = class_names.index(args.aux_negative_class)
            print(
                f"Auxiliary head enabled: {args.aux_positive_class}=1, {args.aux_negative_class}=0, weight={args.aux_loss_weight:.3f}"
            )

    class_weights = parse_class_weights(args.focal_class_weights, len(class_names))
    main_head_loss = build_main_head_loss(args, class_weights)

    if args.main_loss == "focal":
        print(f"Main loss: focal, class weights: {class_weights}")
    else:
        print(f"Main loss: cce, label smoothing: {float(np.clip(args.label_smoothing, 0.0, 0.3)):.3f}")

    target_lab_mean = None
    target_lab_std = None
    if args.color_normalization == "reinhard":
        target_lab_mean, target_lab_std = estimate_reinhard_reference(
            samples=train_samples,
            image_size=args.img_size,
            seed=args.seed,
            max_samples=args.normalization_reference_samples,
        )

        if target_lab_mean is None or target_lab_std is None:
            print("Color normalization reference could not be estimated; continuing without normalization.")
            args.color_normalization = "none"
        else:
            print(
                "Reinhard reference estimated: "
                f"mean={np.round(target_lab_mean, 2).tolist()}, "
                f"std={np.round(target_lab_std, 2).tolist()}"
            )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    write_split_summary(
        summary_path=results_dir / "train_val_split_summary.csv",
        class_names=class_names,
        train_samples=train_samples,
        val_samples=val_samples,
    )

    train_seq = WBCSequence(
        samples=train_samples,
        class_names=class_names,
        batch_size=args.batch_size,
        image_size=args.img_size,
        seed=args.seed,
        training=True,
        enable_aux_binary_head=enable_aux_binary_head,
        aux_positive_index=aux_positive_index,
        aux_negative_index=aux_negative_index,
        color_normalization=args.color_normalization,
        target_lab_mean=target_lab_mean,
        target_lab_std=target_lab_std,
        crop_prob=args.crop_prob,
        bg_randomization_prob=args.bg_randomization_prob,
        bg_randomization_strength=args.bg_randomization_strength,
        stain_jitter_prob=args.stain_jitter_prob,
        neutrophil_aug_scale=args.neutrophil_aug_scale,
        lymphocyte_aug_scale=args.lymphocyte_aug_scale,
    )
    val_seq = WBCSequence(
        samples=val_samples,
        class_names=class_names,
        batch_size=args.batch_size,
        image_size=args.img_size,
        seed=args.seed,
        training=False,
        enable_aux_binary_head=enable_aux_binary_head,
        aux_positive_index=aux_positive_index,
        aux_negative_index=aux_negative_index,
        color_normalization=args.color_normalization,
        target_lab_mean=target_lab_mean,
        target_lab_std=target_lab_std,
        crop_prob=0.0,
        bg_randomization_prob=0.0,
        bg_randomization_strength=0.0,
        stain_jitter_prob=0.0,
        neutrophil_aug_scale=0.0,
        lymphocyte_aug_scale=0.0,
    )

    if args.dry_run:
        dry_batch = train_seq[0]
        if enable_aux_binary_head:
            x_batch, y_batch, sw_batch = dry_batch
            print(
                f"Dry-run batch shapes: x={x_batch.shape}, "
                f"y_main={y_batch[0].shape}, "
                f"y_aux={y_batch[1].shape}, "
                f"aux_weight_active={int(np.sum(sw_batch[1]))}"
            )
        else:
            x_batch, y_batch = dry_batch
            print(f"Dry-run batch shapes: x={x_batch.shape}, y={y_batch.shape}")
        return

    model, base_model = build_model(
        num_classes=len(class_names),
        image_size=args.img_size,
        enable_aux_binary_head=enable_aux_binary_head,
    )

    if enable_aux_binary_head:
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=[
                main_head_loss,
                "binary_crossentropy",
            ],
            loss_weights=[1.0, float(args.aux_loss_weight)],
            metrics=[["accuracy"], ["accuracy"]],
        )
        monitor_metric = "val_main_out_accuracy"
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=main_head_loss,
            metrics=["accuracy"],
        )
        monitor_metric = "val_accuracy"

    rng = np.random.default_rng(args.seed)
    candidate_paths = [path for path, _ in val_samples]
    sample_count = min(args.xai_sample_count, len(candidate_paths))
    xai_sample_paths = list(rng.choice(candidate_paths, size=sample_count, replace=False)) if sample_count else []

    xai_callback = XAIFocusMonitor(
        sample_paths=xai_sample_paths,
        image_size=args.img_size,
        threshold=args.xai_focus_threshold,
        patience=args.xai_patience,
        every_n_epochs=args.xai_every_n_epochs,
        color_normalization=args.color_normalization,
        target_lab_mean=target_lab_mean,
        target_lab_std=target_lab_std,
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.model_path,
            save_best_only=True,
            monitor=monitor_metric,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode="max",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        xai_callback,
    ]

    print("\n" + "=" * 60)
    print("PHASE 1: feature extraction (base frozen)")
    print("=" * 60)
    history_phase1 = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.phase1_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("PHASE 2: fine-tuning (base unfrozen)")
    print("=" * 60)

    base_model.trainable = True

    if enable_aux_binary_head:
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss=[
                main_head_loss,
                "binary_crossentropy",
            ],
            loss_weights=[1.0, float(args.aux_loss_weight)],
            metrics=[["accuracy"], ["accuracy"]],
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss=main_head_loss,
            metrics=["accuracy"],
        )

    history_phase2 = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.phase2_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    plot_history(
        history_phase1,
        history_phase2,
        output_path=results_dir / "training_history_shortcut_resistant.png",
    )

    print(f"\nTraining completed. Best model path: {args.model_path}")
    print(f"Split summary: {results_dir / 'train_val_split_summary.csv'}")
    print(f"History plot: {results_dir / 'training_history_shortcut_resistant.png'}")


if __name__ == "__main__":
    main()
