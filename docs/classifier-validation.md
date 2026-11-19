# LeafGen AI – Classifier Validation Report

## 1. Overview

This report documents four sample tests performed using the trained LeafGen AI image classification model.

The purpose was to verify disease classification across healthy and diseased leaves from tomato and potato plants.

## 2. Test Results

| No. | Test Image | Expected Class | Predicted Class | Confidence | Result |
|---|---|---|---|---:|---|
| 1 | Tomato Early blight | Tomato - Early blight | Tomato - Early blight | 95.29% | Passed |
| 2 | Healthy tomato | Tomato - healthy | Tomato - healthy | 97.35% | Passed |
| 3 | Tomato Late blight | Tomato - Late blight | Tomato - Late blight | 99.99% | Passed |
| 4 | Potato Late blight | Potato - Late blight | Potato - Late blight | 99.43% | Passed |

## 3. Detailed Observations

### Test 1: Tomato Early blight

- Predicted disease: Tomato - Early blight
- Confidence: 95.29%
- Second prediction: Tomato - Late blight (3.85%)
- Third prediction: Tomato - Leaf Mold (0.45%)
- API status: Success
- Warning: None

### Test 2: Healthy tomato leaf

- Predicted class: Tomato - healthy
- Confidence: 97.35%
- Second prediction: Tomato - Late blight (1.09%)
- Third prediction: Tomato - Target Spot (0.50%)
- API status: Success
- Warning: None

### Test 3: Tomato Late blight

- Predicted disease: Tomato - Late blight
- Confidence: 99.99%
- API status: Success
- Warning: None

### Test 4: Potato Late blight

- Predicted disease: Potato - Late blight
- Confidence: 99.43%
- Second prediction: Tomato - Late blight (0.57%)
- API status: Success
- Warning: None

## 4. Summary

- Total sample tests: 4
- Successful expected predictions: 4
- Unsuccessful expected predictions: 0
- API errors: 0
- API warnings: 0

All four tested images received the expected classification.

## 5. Limitations

These results represent four individual sample tests. They do not establish the overall accuracy or generalization performance of the model.

A systematic evaluation using a separate labelled test dataset is required to calculate:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class performance

## 6. Conclusion

The initial sample testing demonstrates that the LeafGen AI classifier can return plausible predictions for healthy and diseased tomato leaves and potato late blight.

Further testing across a larger and independent dataset is required before drawing conclusions about real-world model performance.

