import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.custom_layers import MedSwish, WBCAttentionBlock
from src.custom_losses import WBCFocalLoss
from src.preprocessing import PreprocessingFilters


CLASS_NAMES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="TestA ve TestB klasorleri icin classification report ve confusion matrix uretir."
    )
    parser.add_argument(
        "--data-root",
        default="data/raabin-wbc-data",
        help="TestA ve TestB klasorlerini iceren ana dizin.",
    )
    parser.add_argument(
        "--model-path",
        default="data/models/wbc_final_model_densenet.keras",
        help="Keras model dosya yolu. Verilmezse yaygin konumlar otomatik denenir.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/final",
        help="Rapor ve gorsellerin kaydedilecegi dizin.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch boyutu.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Hizli test icin her split'te islenecek maksimum goruntu sayisi.",
    )
    parser.add_argument(
        "--tta",
        choices=["none", "light"],
        default="light",
        help="Test-time augmentation modu. 'light' modunda flip/rotasyon/parlaklik ortalamasi uygulanir.",
    )
    parser.add_argument(
        "--testb-binary-mode",
        choices=["none", "main", "aux"],
        default="main",
        help="Sadece TestB icin 2 sinifa zorlayici tahmin modu: none=5-sinif, main=main_out icinden N/L sec, aux=aux cikisindan N/L sec.",
    )
    parser.add_argument(
        "--color-normalization",
        choices=["none", "reinhard"],
        default="reinhard",
        help="Inference oncesi renk normalizasyonu.",
    )
    parser.add_argument(
        "--normalization-reference-split",
        default="Train",
        help="Reinhard referans istatistiklerinin hesaplanacagi split klasoru.",
    )
    parser.add_argument(
        "--normalization-reference-samples",
        type=int,
        default=1200,
        help="Reinhard referansi icin kullanilacak maksimum goruntu sayisi.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Rastgelelik tohumu.",
    )
    return parser.parse_args()


def resolve_model_path(explicit_path=None):
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model bulunamadi: {candidate}")

    default_path = Path("data/models/wbc_final_model_densenet.keras")
    if default_path.exists():
        return default_path

    raise FileNotFoundError(f"Model bulunamadi: {default_path}")


def load_trained_model(model_path):
    custom_objects = {
        "MedSwish": MedSwish,
        "WBCAttentionBlock": WBCAttentionBlock,
        "WBCFocalLoss": WBCFocalLoss,
        "wbc_focal_loss": WBCFocalLoss,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def extract_predictions(predictions):
    aux_predictions = None

    if isinstance(predictions, dict):
        if "main_out" in predictions:
            main_predictions = predictions["main_out"]
        else:
            main_predictions = next(iter(predictions.values()))

        if "aux_binary_out" in predictions:
            aux_predictions = predictions["aux_binary_out"]
    elif isinstance(predictions, (list, tuple)):
        main_predictions = predictions[0]
        if len(predictions) > 1:
            aux_predictions = predictions[1]
    else:
        main_predictions = predictions

    main_predictions = np.array(main_predictions, dtype=np.float32)
    if aux_predictions is None:
        return main_predictions, None

    aux_predictions = np.array(aux_predictions, dtype=np.float32).reshape((-1,))
    return main_predictions, aux_predictions


def estimate_reinhard_reference(data_root, split_name, image_size, max_samples, seed):
    split_dir = Path(data_root) / split_name
    if not split_dir.exists():
        print(f"Uyari: Reinhard referans split bulunamadi: {split_dir}")
        return None, None

    image_paths = []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)

    if not image_paths:
        print("Uyari: Reinhard referansi icin goruntu bulunamadi.")
        return None, None

    rng = np.random.default_rng(seed)
    n_samples = min(max_samples, len(image_paths))
    chosen = list(rng.choice(image_paths, size=n_samples, replace=False))

    means = []
    stds = []
    for image_path in chosen:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size))
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        means.append(np.mean(lab, axis=(0, 1)))
        stds.append(np.std(lab, axis=(0, 1)) + 1e-6)

    if not means:
        print("Uyari: Reinhard referans istatistigi hesaplanamadi.")
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


def collect_samples(split_dir, limit=None):
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split klasoru bulunamadi: {split_path}")

    available_classes = [class_name for class_name in CLASS_NAMES if (split_path / class_name).exists()]
    unknown_dirs = [
        path.name for path in split_path.iterdir() if path.is_dir() and path.name not in CLASS_NAMES
    ]

    if unknown_dirs:
        print(f"Uyari: taninmayan sinif klasorleri atlandi: {', '.join(sorted(unknown_dirs))}")

    if not available_classes:
        raise ValueError(f"Desteklenen sinif klasoru bulunamadi: {split_path}")

    samples = []
    for class_name in available_classes:
        class_dir = split_path / class_name

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append((image_path, class_name))
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


def preprocess_image(image_path, color_normalization, target_lab_mean, target_lab_std):
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Goruntu okunamadi: {image_path}") from exc

    image_np = np.array(image)
    image_np = cv2.resize(image_np, (224, 224))
    if color_normalization == "reinhard":
        image_np = apply_reinhard_normalization(image_np, target_lab_mean, target_lab_std)
    return PreprocessingFilters.medical_enhanced(image_np)


def build_tta_batch(image):
    # Keep TTA deterministic for reproducible evaluation.
    augmented = [
        image,
        np.fliplr(image),
        np.flipud(image),
        np.rot90(image, 1),
        np.rot90(image, 2),
        np.rot90(image, 3),
        np.clip(image * 1.1, 0.0, 1.0),
        np.clip(image * 0.9, 0.0, 1.0),
    ]
    return np.array(augmented, dtype=np.float32)


def infer_main_and_aux(model, batch_images, tta_mode):
    if tta_mode == "none":
        predictions = model.predict(np.array(batch_images, dtype=np.float32), verbose=0)
        return extract_predictions(predictions)

    aggregated_main = []
    aggregated_aux = []
    aux_found = False

    for image in batch_images:
        tta_batch = build_tta_batch(image)
        predictions = model.predict(tta_batch, verbose=0)
        main_predictions, aux_predictions = extract_predictions(predictions)
        aggregated_main.append(np.mean(main_predictions, axis=0))

        if aux_predictions is not None:
            aux_found = True
            aggregated_aux.append(float(np.mean(aux_predictions)))

    main_out = np.array(aggregated_main, dtype=np.float32)
    if not aux_found:
        return main_out, None

    return main_out, np.array(aggregated_aux, dtype=np.float32)


def apply_testb_binary_mode(main_predictions, aux_predictions, mode):
    lymphocyte_idx = CLASS_NAMES.index("Lymphocyte")
    neutrophil_idx = CLASS_NAMES.index("Neutrophil")

    if mode == "none":
        predicted_indices = np.argmax(main_predictions, axis=1)
        return predicted_indices, main_predictions

    if mode == "main":
        binary_probs = main_predictions[:, [lymphocyte_idx, neutrophil_idx]]
        denom = np.sum(binary_probs, axis=1, keepdims=True) + 1e-8
        binary_probs = binary_probs / denom
        pick_neutrophil = binary_probs[:, 1] >= binary_probs[:, 0]
        predicted_indices = np.where(pick_neutrophil, neutrophil_idx, lymphocyte_idx)

        adjusted_probs = np.zeros_like(main_predictions)
        adjusted_probs[:, lymphocyte_idx] = binary_probs[:, 0]
        adjusted_probs[:, neutrophil_idx] = binary_probs[:, 1]
        return predicted_indices, adjusted_probs

    if aux_predictions is None:
        print("Uyari: aux_binary_out bulunamadi, testb-binary-mode=aux modu main moduna dusuruldu.")
        return apply_testb_binary_mode(main_predictions, aux_predictions, mode="main")

    aux_probs = np.clip(aux_predictions.astype(np.float32), 0.0, 1.0)
    predicted_indices = np.where(aux_probs >= 0.5, neutrophil_idx, lymphocyte_idx)

    adjusted_probs = np.zeros_like(main_predictions)
    adjusted_probs[:, lymphocyte_idx] = 1.0 - aux_probs
    adjusted_probs[:, neutrophil_idx] = aux_probs
    return predicted_indices, adjusted_probs


def predict_samples(
    model,
    samples,
    batch_size,
    split_name,
    tta_mode,
    testb_binary_mode,
    color_normalization,
    target_lab_mean,
    target_lab_std,
):
    label_to_index = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y_true = []
    y_pred = []
    probabilities = []
    file_paths = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        batch_images = []
        batch_true = []
        batch_paths = []

        for image_path, class_name in batch:
            try:
                batch_images.append(
                    preprocess_image(
                        image_path,
                        color_normalization=color_normalization,
                        target_lab_mean=target_lab_mean,
                        target_lab_std=target_lab_std,
                    )
                )
                batch_true.append(label_to_index[class_name])
                batch_paths.append(str(image_path))
            except ValueError as exc:
                print(f"Uyari: {exc}")

        if not batch_images:
            continue

        predictions, aux_predictions = infer_main_and_aux(model, batch_images, tta_mode=tta_mode)

        active_binary_mode = testb_binary_mode if split_name == "TestB" else "none"
        predicted_indices, adjusted_predictions = apply_testb_binary_mode(
            predictions,
            aux_predictions,
            mode=active_binary_mode,
        )

        y_true.extend(batch_true)
        y_pred.extend(predicted_indices.tolist())
        probabilities.extend(adjusted_predictions.tolist())
        file_paths.extend(batch_paths)

    return y_true, y_pred, probabilities, file_paths


def compute_confusion_matrix(y_true, y_pred):
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for true_idx, pred_idx in zip(y_true, y_pred):
        matrix[true_idx, pred_idx] += 1
    return matrix


def build_classification_report(y_true, y_pred):
    matrix = compute_confusion_matrix(y_true, y_pred)
    supports = matrix.sum(axis=1)
    total = int(supports.sum())

    rows = []
    precisions = []
    recalls = []
    f1_scores = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[class_index, class_index]
        predicted_positive = matrix[:, class_index].sum()
        actual_positive = supports[class_index]

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        rows.append((class_name, precision, recall, f1_score, int(actual_positive)))

    accuracy = float(np.trace(matrix) / total) if total else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    if total:
        weighted_precision = float(np.average(precisions, weights=supports))
        weighted_recall = float(np.average(recalls, weights=supports))
        weighted_f1 = float(np.average(f1_scores, weights=supports))
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    lines = []
    lines.append(f"{'class':<15}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>10}")
    lines.append("")
    for class_name, precision, recall, f1_score, support in rows:
        lines.append(
            f"{class_name:<15}{precision:>12.4f}{recall:>12.4f}{f1_score:>12.4f}{support:>10d}"
        )

    lines.append("")
    lines.append(f"{'accuracy':<15}{'':>12}{'':>12}{accuracy:>12.4f}{total:>10d}")
    lines.append(
        f"{'macro avg':<15}{macro_precision:>12.4f}{macro_recall:>12.4f}{macro_f1:>12.4f}{total:>10d}"
    )
    lines.append(
        f"{'weighted avg':<15}{weighted_precision:>12.4f}{weighted_recall:>12.4f}{weighted_f1:>12.4f}{total:>10d}"
    )

    return "\n".join(lines), matrix


def save_confusion_matrix(y_true, y_pred, output_path, title):
    matrix = compute_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Tahmin",
        ylabel="Gercek",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                matrix[row, col],
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_predictions_csv(output_path, file_paths, y_true, y_pred, probabilities):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        header = ["file_path", "true_label", "pred_label"] + [
            f"prob_{class_name}" for class_name in CLASS_NAMES
        ]
        handle.write(",".join(header) + "\n")

        for file_path, true_idx, pred_idx, probs in zip(file_paths, y_true, y_pred, probabilities):
            row = [
                file_path,
                CLASS_NAMES[true_idx],
                CLASS_NAMES[pred_idx],
            ] + [f"{float(prob):.6f}" for prob in probs]
            handle.write(",".join(row) + "\n")


def evaluate_split(
    model,
    split_name,
    split_dir,
    output_dir,
    batch_size,
    tta_mode,
    testb_binary_mode,
    color_normalization,
    target_lab_mean,
    target_lab_std,
    limit=None,
):
    samples = collect_samples(split_dir, limit=limit)
    if not samples:
        print(f"{split_name}: islenecek goruntu bulunamadi.")
        return None

    print(f"{split_name}: {len(samples)} goruntu isleniyor...")
    y_true, y_pred, probabilities, file_paths = predict_samples(
        model,
        samples,
        batch_size,
        split_name=split_name,
        tta_mode=tta_mode,
        testb_binary_mode=testb_binary_mode,
        color_normalization=color_normalization,
        target_lab_mean=target_lab_mean,
        target_lab_std=target_lab_std,
    )

    report_text, _ = build_classification_report(y_true, y_pred)

    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    report_path = split_output_dir / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    csv_path = split_output_dir / "predictions.csv"
    save_predictions_csv(csv_path, file_paths, y_true, y_pred, probabilities)

    matrix_path = split_output_dir / "confusion_matrix.png"
    save_confusion_matrix(
        y_true,
        y_pred,
        matrix_path,
        title=f"{split_name} Confusion Matrix",
    )

    print(f"\n===== {split_name} =====")
    print(report_text)
    print(f"Rapor: {report_path}")
    print(f"CSV: {csv_path}")
    print(f"Matrix: {matrix_path}\n")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "file_paths": file_paths,
    }


def evaluate_combined(results, output_dir):
    combined_true = []
    combined_pred = []
    combined_probs = []
    combined_paths = []

    for result in results:
        if not result:
            continue
        combined_true.extend(result["y_true"])
        combined_pred.extend(result["y_pred"])
        combined_probs.extend(result["probabilities"])
        combined_paths.extend(result["file_paths"])

    if not combined_true:
        return

    report_text, _ = build_classification_report(combined_true, combined_pred)

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    report_path = combined_dir / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    csv_path = combined_dir / "predictions.csv"
    save_predictions_csv(csv_path, combined_paths, combined_true, combined_pred, combined_probs)

    matrix_path = combined_dir / "confusion_matrix.png"
    save_confusion_matrix(
        combined_true,
        combined_pred,
        matrix_path,
        title="TestA + TestB Confusion Matrix",
    )

    print("\n===== COMBINED =====")
    print(report_text)
    print(f"Rapor: {report_path}")
    print(f"CSV: {csv_path}")
    print(f"Matrix: {matrix_path}\n")


def main():
    args = parse_args()

    model_path = resolve_model_path(args.model_path)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    print(f"Model: {model_path}")
    print(f"Veri klasoru: {data_root}")
    print(f"Cikti klasoru: {output_dir}")
    print(f"TTA: {args.tta}")
    print(f"TestB binary mode: {args.testb_binary_mode}")
    print(f"Color normalization: {args.color_normalization}")

    model = load_trained_model(model_path)

    target_lab_mean = None
    target_lab_std = None
    if args.color_normalization == "reinhard":
        target_lab_mean, target_lab_std = estimate_reinhard_reference(
            data_root=data_root,
            split_name=args.normalization_reference_split,
            image_size=224,
            max_samples=args.normalization_reference_samples,
            seed=args.seed,
        )
        if target_lab_mean is None or target_lab_std is None:
            print("Uyari: Reinhard referansi hesaplanamadi, color-normalization=none olarak devam ediliyor.")
            args.color_normalization = "none"
        else:
            print(
                "Reinhard reference estimated: "
                f"mean={np.round(target_lab_mean, 2).tolist()}, "
                f"std={np.round(target_lab_std, 2).tolist()}"
            )

    results = []
    for split_name in ["TestA", "TestB"]:
        split_dir = data_root / split_name
        results.append(
            evaluate_split(
                model=model,
                split_name=split_name,
                split_dir=split_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                tta_mode=args.tta,
                testb_binary_mode=args.testb_binary_mode,
                color_normalization=args.color_normalization,
                target_lab_mean=target_lab_mean,
                target_lab_std=target_lab_std,
                limit=args.limit,
            )
        )

    evaluate_combined(results, output_dir)


if __name__ == "__main__":
    main()
