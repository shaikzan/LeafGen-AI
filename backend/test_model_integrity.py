"""Executable integrity checks for the shipped trained checkpoints."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from fastapi import HTTPException
from io import BytesIO
from PIL import Image

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

from models.leafgen_model import (  # noqa: E402
    NC, classes, rare, classify, generate, load_classifier, load_generator, predict_proba,
)
from app.services.classifier import predict_image  # noqa: E402
from app.services.generator import generate_leaf_images  # noqa: E402
from evaluate_classifier import evaluate  # noqa: E402


class ModelIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = load_classifier("real")
        cls.augmented_classifier = load_classifier("aug")
        cls.generator = load_generator()

    def test_class_mapping_is_38_unique_entries(self):
        self.assertEqual(NC, 38)
        self.assertEqual(len(classes), 38)
        self.assertEqual(len(set(classes)), 38)
        self.assertEqual(classes[0], "Apple___Apple_scab")
        self.assertEqual(classes[31], "Tomato___Leaf_Mold")
        self.assertEqual([classes[i] for i in rare], [
            "Apple___Cedar_apple_rust", "Tomato___Tomato_mosaic_virus",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Apple___Black_rot",
            "Apple___Apple_scab", "Tomato___Leaf_Mold",
        ])

    def test_classifier_has_loaded_38_class_resnet18_and_eval_mode(self):
        self.assertEqual(tuple(self.classifier.fc.weight.shape), (38, 512))
        self.assertFalse(self.classifier.training)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.classifier.parameters()))
        self.assertEqual(tuple(self.augmented_classifier.fc.weight.shape), (38, 512))
        self.assertFalse(self.augmented_classifier.training)
        self.assertFalse(torch.equal(self.classifier.fc.weight, self.augmented_classifier.fc.weight))

    def test_inference_returns_finite_normalized_probabilities_and_top_three(self):
        # Synthetic input checks inference plumbing, not disease accuracy.
        rgb = np.full((256, 192, 3), 127, dtype=np.uint8)
        probabilities = predict_proba(self.classifier, rgb)
        self.assertEqual(probabilities.shape, (38,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
        top = classify(self.classifier, rgb, k=3)
        self.assertEqual(len(top), 3)
        self.assertEqual(len({name for name, _ in top}), 3)
        self.assertAlmostEqual(sum(score for _, score in top), float(sum(probabilities[i] for i in probabilities.argsort()[-3:])), places=6)
        self.assertEqual(top, classify(self.classifier, rgb, k=3))

    def test_invalid_classifier_inputs_fail_clearly(self):
        with self.assertRaises(ValueError):
            classify(self.classifier, np.zeros((32, 32), dtype=np.uint8))
        with self.assertRaises(ValueError):
            classify(self.classifier, np.zeros((32, 32, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            classify(self.classifier, np.zeros((32, 32, 3), dtype=np.uint8), k=0)

    def test_generator_is_conditioned_and_returns_valid_diverse_images(self):
        # Hold noise constant to establish that the label input changes output.
        z = torch.randn(1, 100).repeat(2, 1)
        y = torch.tensor([classes.index("Apple___Apple_scab"), classes.index("Tomato___Late_blight")])
        conditioned = self.generator(z, y)
        self.assertGreater(float(torch.mean(torch.abs(conditioned[0] - conditioned[1])).detach()), 1e-6)

        classes_to_check = [
            "Apple___Apple_scab", "Grape___Black_rot", "Tomato___Late_blight",
            "Strawberry___Leaf_scorch", "Apple___healthy",
        ]
        samples = []
        for class_code in classes_to_check:
            batch = generate(self.generator, class_code, n=2)
            self.assertEqual(batch.shape, (2, 64, 64, 3))
            self.assertEqual(batch.dtype, np.uint8)
            self.assertGreater(int(batch.max()), int(batch.min()))
            self.assertGreater(float(np.mean(np.abs(batch[0].astype(np.int16)-batch[1].astype(np.int16)))), 0.1)
            samples.append(batch[0])
        differences = [float(np.mean(np.abs(a.astype(np.int16)-b.astype(np.int16)))) for i,a in enumerate(samples) for b in samples[i+1:]]
        self.assertTrue(any(d > 0.1 for d in differences), "Class conditions produced identical images")

    def test_generator_rejects_unknown_classes(self):
        with self.assertRaises(ValueError):
            generate(self.generator, "unknown___class", n=1)

    def test_prediction_service_valid_and_corrupt_image_handling(self):
        buffer = BytesIO()
        Image.new("RGB", (64, 64), (80, 130, 40)).save(buffer, format="PNG")
        result = predict_image(self.classifier, buffer.getvalue())
        self.assertTrue(result["success"])
        self.assertEqual(len(result["top_predictions"]), 3)
        self.assertIn("predicted_crop", result)
        self.assertIn("predicted_disease", result)
        self.assertEqual(result["score_type"], "uncalibrated_softmax")
        self.assertIsNone(result["calibrated_score"])
        self.assertEqual(result["input_validity_status"], "not_checked")
        self.assertTrue(0 <= result["confidence"] <= 1)
        self.assertLessEqual(sum(x["confidence"] for x in result["top_predictions"]), 1.0)
        for invalid in (b"", b"not an image"):
            with self.assertRaises(HTTPException) as err:
                predict_image(self.classifier, invalid)
            self.assertEqual(err.exception.status_code, 400)

    def test_prediction_rejects_oversized_decoded_dimensions(self):
        # A valid PNG header is enough to exercise the dimension guard before
        # decoding pixel data or allocating a large image.
        import struct
        import zlib
        width, height = 5001, 5000
        def png_chunk(tag, data):
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(b"")) + png_chunk(b"IEND", b"")
        with self.assertRaises(HTTPException) as err:
            predict_image(self.classifier, png)
        self.assertEqual(err.exception.status_code, 413)

    def test_generation_service_returns_actual_encoded_pngs(self):
        result = generate_leaf_images(self.generator, "Apple___Apple_scab", 2)
        self.assertTrue(result["success"])
        self.assertEqual(result["number_of_images"], 2)
        self.assertEqual([image["width"] for image in result["images"]], [64, 64])
        self.assertTrue(all(image["mime_type"] == "image/png" and image["data_base64"] for image in result["images"]))

    def test_evaluation_script_writes_reports_for_labeled_folder(self):
        # This checks evaluator plumbing only; the synthetic image is not
        # accuracy evidence and is intentionally not reported as a benchmark.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "labels"
            folder = root / "Apple___Apple_scab"
            folder.mkdir(parents=True)
            Image.new("RGB", (48, 48), (70, 120, 35)).save(folder / "smoke.png")
            output = Path(temp) / "report"
            summary = evaluate(root, output, "smoke_test")
            self.assertEqual(summary["disease_labeled_images"], 1)
            self.assertEqual(summary["input_validity_detector"], "not implemented; input validity is not evaluated")
            for name in ("predictions.csv", "summary.json", "confusion_matrix.csv", "per_class_metrics.csv", "prediction_distribution.csv"):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
