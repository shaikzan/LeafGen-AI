"""Evaluate the shipped classifier against labeled folders without data leakage.

Expected layout: DATA_ROOT/<exact-class-code>/*.(jpg|jpeg|png). For separate
external testing, pass a different directory and label the report as external.
An optional __non_leaf__ directory is recorded as an unsupported-input control;
it is not counted as a disease-class prediction metric because no validity
detector is currently available.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))
from models.leafgen_model import classes, classify, load_classifier  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def evaluate(data_root: Path, output_dir: Path, split_name: str = "held_out"):
    if not data_root.is_dir():
        raise FileNotFoundError(f"Labeled image directory does not exist: {data_root}")
    known = set(classes)
    labeled = []
    for folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if folder.name != "__non_leaf__" and folder.name not in known:
            raise ValueError(f"Unknown label folder {folder.name!r}; use exact class codes from classes.json.")
        for image_path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            labeled.append((image_path, folder.name))
    if not labeled:
        raise ValueError(f"No labeled JPG/PNG images found under {data_root}.")

    model = load_classifier("real")
    rows = []
    class_counts = Counter()
    class_correct = Counter()
    class_top3 = Counter()
    prediction_counts = Counter()
    true_positives = Counter()
    false_positives = Counter()
    false_negatives = Counter()
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    total_leaf = correct = top3_correct = 0
    nonleaf_count = 0

    for image_path, expected in labeled:
        try:
            with Image.open(image_path) as source:
                rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            rows.append({"filename": str(image_path.relative_to(data_root)), "expected_class": expected,
                         "predicted_class": "", "top_three": "[]", "raw_model_score": "",
                         "correct": False, "input_validity": "not_checked", "error": f"Invalid image: {exc}"})
            continue
        top = classify(model, rgb, k=3)
        predicted = top[0][0]
        is_leaf_label = expected != "__non_leaf__"
        rows.append({
            "filename": str(image_path.relative_to(data_root)), "expected_class": expected,
            "predicted_class": predicted,
            "top_three": json.dumps([{"class_code": c, "raw_softmax_score": float(s)} for c, s in top]),
            "raw_model_score": float(top[0][1]),
            "correct": bool(is_leaf_label and predicted == expected),
            "input_validity": "not_available_no_detector",
            "error": "" if is_leaf_label else "Non-leaf control: no validity detector; disease output is not evaluated.",
        })
        if not is_leaf_label:
            nonleaf_count += 1
            continue
        total_leaf += 1
        class_counts[expected] += 1
        prediction_counts[predicted] += 1
        is_correct = predicted == expected
        correct += int(is_correct)
        class_correct[expected] += int(is_correct)
        class_top3[expected] += int(expected in [code for code, _ in top])
        confusion[classes.index(expected), classes.index(predicted)] += 1
        if is_correct:
            true_positives[expected] += 1
        else:
            false_positives[predicted] += 1
            false_negatives[expected] += 1

    recalls = {c: class_correct[c] / n for c, n in class_counts.items()}
    f1_values = []
    precisions = {}
    for c, support in class_counts.items():
        precision = true_positives[c] / (true_positives[c] + false_positives[c]) if true_positives[c] + false_positives[c] else 0.0
        precisions[c] = precision
        recall = class_correct[c] / support
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    summary = {
        "split": split_name,
        "data_root": str(data_root.resolve()),
        "disease_labeled_images": total_leaf,
        "non_leaf_controls": nonleaf_count,
        "invalid_files": sum(bool(row["error"].startswith("Invalid image")) for row in rows),
        "accuracy_top1": correct / total_leaf if total_leaf else None,
        "accuracy_top3": top3_correct / total_leaf if total_leaf else None,
        "macro_f1_over_present_classes": float(np.mean(f1_values)) if f1_values else None,
        "macro_precision_over_present_classes": float(np.mean(list(precisions.values()))) if precisions else None,
        "macro_recall_over_present_classes": float(np.mean(list(recalls.values()))) if recalls else None,
        "input_validity_detector": "not implemented; input validity is not evaluated",
        "score_type": "uncalibrated softmax score",
        "class_support": dict(class_counts),
        "per_class_recall": recalls,
        "prediction_distribution": dict(prediction_counts),
    }
    # Top-3 count is tracked per class; the aggregate is reconstructed from rows.
    top3_correct = sum(1 for row in rows if row["expected_class"] in known and row["expected_class"] in [p["class_code"] for p in json.loads(row["top_three"])])
    summary["accuracy_top3"] = top3_correct / total_leaf if total_leaf else None

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["expected\\predicted", *classes])
        for i, code in enumerate(classes):
            writer.writerow([code, *confusion[i].tolist()])
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_code", "support", "precision", "recall", "top3_recall"])
        for code in classes:
            if class_counts[code]:
                writer.writerow([code, class_counts[code], precisions[code], recalls[code], class_top3[code] / class_counts[code]])
    with (output_dir / "prediction_distribution.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["predicted_class", "count"])
        writer.writerows(sorted(prediction_counts.items()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Labeled image directory, one exact class-code subfolder per class")
    parser.add_argument("--output", type=Path, required=True, help="Directory for CSV and JSON evaluation reports")
    parser.add_argument("--split", choices=("held_out", "external"), default="held_out")
    args = parser.parse_args()
    try:
        summary = evaluate(args.data_root, args.output, args.split)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
