# WBC Analyzer — Multi-modal Agentic AI for White Blood Cell Classification

**BSc Final Project — Sakarya University, Information Systems Engineering, 2026**

## About

An end-to-end, multi-modal deep learning system for automated white blood cell classification from peripheral blood smear images. Built with a custom DenseNet121 architecture, CBAM attention mechanisms, and Grad-CAM explainability.

Taking inspiration from modern Agentic AI workflows, the system integrates an **Autonomous LLM Hematologist** (Gemini 2.5 Flash) that analyzes the CNN's spatial reasoning (heatmaps) to generate real-time clinical reports and detect "Shortcut Learning" phenomena.

## Features

- **DenseNet121 + CBAM Attention Block:** Custom architecture for robust WBC classification.
- **Agentic XAI Reporting (Multi-modal):** Integrates Gemini 2.5 Flash Vision to autonomously interpret Grad-CAM heatmaps and validate model focus.
- **Shortcut Learning Detection:** The LLM agent automatically detects if the model is "cheating" by focusing on the background rather than cellular morphology.
- **Medical Enhanced Filter:** CLAHE + bilateral filtering + selective sharpening preprocessing pipeline.
- **Custom Loss & Activation:** WBC Focal Loss + MedSwish for imbalanced medical datasets.
- **Flask Web Interface:** Real-time drag-and-drop image analysis with interactive UI.
- **95.12% Combined Test Accuracy** with domain-shift robust inference (TestA: 98.53%, TestB: 88.15%).

## Demo

_in progress..._

## Model Access

The trained model file (`.keras`) is not included in this repository due to its size.  
📥 **Download:** [wbc_final_model_densenet.keras](https://drive.google.com/file/d/1imbnTiTTxEpuXD_HB_IJm0RxE03PJ6GQ/view?usp=sharing)  
After downloading, place it in: `data/models/`

## Quick Start

```bash
git clone https://github.com/frissonitte/wbc-analyzer-final.git
cd wbc-final
pip install -r requirements.txt
```

Set your Gemini API key in a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

Then run the app:

```bash
python app.py
```

## Final Inference (No Retraining)

Recommended evaluation command reproducing the best reported results:

```bash
python class.py \
    --model-path data/models/wbc_final_model_densenet.keras \
    --data-root data/raabin-wbc-data \
    --output-dir outputs/final \
    --testb-binary-mode main \
    --tta light \
    --color-normalization reinhard
```

### Ablation summary

> **Note:** TestB contains only Lymphocyte and Neutrophil classes (different lab/acquisition conditions from TestA), making it a natural out-of-distribution benchmark for domain generalization.

| Method                      |  TestA |  TestB | Combined |
| --------------------------- | -----: | -----: | -------: |
| Baseline (no adaptation)    | 97.46% | 56.96% |   84.17% |
| + Binary routing            | 97.46% | 73.90% |   89.73% |
| + Reinhard normalization    | 97.99% | 86.46% |   94.21% |
| + TTA (final configuration) | 98.53% | 88.15% |   95.12% |

### Per-class results (final configuration)

**TestA** (n = 4,339):

| Class            |  Precision |     Recall |   F1-score |  Support |
| ---------------- | ---------: | ---------: | ---------: | -------: |
| Basophil         |     1.0000 |     1.0000 |     1.0000 |       89 |
| Eosinophil       |     0.9398 |     0.9689 |     0.9541 |      322 |
| Lymphocyte       |     0.9809 |     0.9923 |     0.9865 |     1034 |
| Monocyte         |     0.9437 |     0.9316 |     0.9376 |      234 |
| Neutrophil       |     0.9958 |     0.9887 |     0.9923 |     2660 |
| **weighted avg** | **0.9854** | **0.9853** | **0.9853** | **4339** |

**TestB** (n = 2,119 — Lymphocyte & Neutrophil only):

| Class            |  Precision |     Recall |   F1-score |  Support |
| ---------------- | ---------: | ---------: | ---------: | -------: |
| Lymphocyte       |     0.3696 |     0.9865 |     0.5378 |      148 |
| Neutrophil       |     0.9988 |     0.8737 |     0.9321 |     1971 |
| **weighted avg** | **0.9549** | **0.8815** | **0.9045** | **2119** |

**Combined** (n = 6,458):

| Class            |  Precision |     Recall |   F1-score |  Support |
| ---------------- | ---------: | ---------: | ---------: | -------: |
| Basophil         |     1.0000 |     1.0000 |     1.0000 |       89 |
| Eosinophil       |     0.9398 |     0.9689 |     0.9541 |      322 |
| Lymphocyte       |     0.8133 |     0.9915 |     0.8936 |     1182 |
| Monocyte         |     0.9437 |     0.9316 |     0.9376 |      234 |
| Neutrophil       |     0.9970 |     0.9398 |     0.9675 |     4631 |
| **weighted avg** | **0.9587** | **0.9512** | **0.9527** | **6458** |

## Shortcut-Resistant Retraining

Use this pipeline to reduce background shortcut learning.

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

What this script does:

- Validation split is created from Train (TestA/TestB remain untouched for final evaluation).
- Foreground-focused crop is applied during training.
- Background randomization is applied during training.
- Multi-task head is enabled by default with `main_out` (5-class softmax) for final prediction.
- Auxiliary output `aux_binary_out` adds Neutrophil-vs-Lymphocyte supervision for morphology-aware regularization.
- XAI foreground-focus ratio is monitored each epoch and can early-stop training when attention drifts to background.

## Roadmap

- [x] DenseNet121 + Attention Block model
- [x] Medical Enhanced preprocessing
- [x] Flask web interface
- [x] Grad-CAM explainability (XAI)
- [x] Multi-modal LLM integration (Agentic reporting)
- [x] Domain-shift robust inference (binary routing + Reinhard normalization + light TTA)

## Previous Version

This project is an extension of the ISE401 design project:  
👉 [WBC-Classification-Project](https://github.com/frissonitte/WBC-Classification-Project)

---

Emirhan Yıldırım — emirhan.yildirim2@ogr.sakarya.edu.tr  
Sakarya University, Information Systems Engineering
