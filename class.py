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
        default=None,
        help="Keras model dosya yolu. Verilmezse yaygin konumlar otomatik denenir.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/classification_metrics",
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
    return parser.parse_args()


def resolve_model_path(explicit_path=None):
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model bulunamadi: {candidate}")

    candidates = [
        Path("models/wbc_final_model_densenet.keras"),
        Path("data/models/wbc_final_model_densenet.keras"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Model bulunamadi. Denenen yollar: {searched}")


def load_trained_model(model_path):
    custom_objects = {
        "MedSwish": MedSwish,
        "WBCAttentionBlock": WBCAttentionBlock,
        "WBCFocalLoss": WBCFocalLoss,
        "wbc_focal_loss": WBCFocalLoss,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


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


def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Goruntu okunamadi: {image_path}") from exc

    image_np = np.array(image)
    image_np = cv2.resize(image_np, (224, 224))
    return PreprocessingFilters.medical_enhanced(image_np)


def predict_samples(model, samples, batch_size):
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
                batch_images.append(preprocess_image(image_path))
                batch_true.append(label_to_index[class_name])
                batch_paths.append(str(image_path))
            except ValueError as exc:
                print(f"Uyari: {exc}")

        if not batch_images:
            continue

        predictions = model.predict(np.array(batch_images, dtype=np.float32), verbose=0)
        predicted_indices = np.argmax(predictions, axis=1)

        y_true.extend(batch_true)
        y_pred.extend(predicted_indices.tolist())
        probabilities.extend(predictions.tolist())
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


def evaluate_split(model, split_name, split_dir, output_dir, batch_size, limit=None):
    samples = collect_samples(split_dir, limit=limit)
    if not samples:
        print(f"{split_name}: islenecek goruntu bulunamadi.")
        return None

    print(f"{split_name}: {len(samples)} goruntu isleniyor...")
    y_true, y_pred, probabilities, file_paths = predict_samples(model, samples, batch_size)

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

    model = load_trained_model(model_path)

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
                limit=args.limit,
            )
        )

    evaluate_combined(results, output_dir)


if __name__ == "__main__":
    main()
