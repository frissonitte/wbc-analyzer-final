# WBC Analyzer: AI-Powered Pathology Assistant

<p align="center">
  <img src="banner.png" alt="WBC Analyzer Banner" width="100%">
</p>

Multi-modal agentic AI for white blood cell classification.

This repository contains the English project documentation. The Turkish version is available in [README.tr.md](README.tr.md).

## Overview

An end-to-end deep learning system for automated white blood cell classification from peripheral blood smear images. The project uses a custom DenseNet121 architecture, CBAM attention blocks, Grad-CAM explainability, and an LLM reporting workflow for XAI summaries.

## Key Features

- DenseNet121 + CBAM attention for WBC classification
- Grad-CAM-based explainability with foreground-masked visualization
- Agentic reporting for heatmap interpretation:
  Primary model: OpenAI GPT-4o (GitHub Models)
  Secondary fallback: Gemini 2.5 Flash
- Medical preprocessing pipeline with CLAHE, bilateral filtering, and selective sharpening
- Custom loss and activation functions for imbalanced medical data
- Flask web interface for drag-and-drop inference

## LLM Reporting Order

1. Primary: `openai/gpt-4o` via GitHub Models (`GITHUB_TOKEN`)
2. Secondary fallback: `gemini-2.5-flash` (`GEMINI_API_KEY`)

## Model Access

The trained `.keras` model is not included in the repository because of its size.

[Download the model](https://drive.google.com/file/d/1imbnTiTTxEpuXD_HB_IJm0RxE03PJ6GQ/view?usp=sharing) and place it in `data/models/` as `wbc_final_model_densenet.keras`.

## Quick Start

```bash
git clone https://github.com/frissonitte/wbc-analyzer-final.git
cd wbc-final
pip install -r requirements.txt
```

Create a `.env` file with your model keys:

```env
GITHUB_TOKEN=your_github_models_token
GEMINI_API_KEY=your_gemini_api_key
```

Run the web app:

```bash
python app.py
```

## Final Inference

To reproduce the best reported results without retraining:

```bash
python class.py \
    --model-path data/models/wbc_final_model_densenet.keras \
    --data-root data/raabin-wbc-data \
    --output-dir outputs/final \
    --testb-binary-mode main \
    --tta light \
    --color-normalization reinhard
```

## Shortcut-Resistant Retraining

Use the retraining pipeline to reduce background shortcut learning:

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

## Repository Notes

- Main project documentation: [README.md](README.md)
- Previous project: [WBC-Classification-Project](https://github.com/frissonitte/WBC-Classification-Project)

## Author

Emirhan Yıldırım  
emirhan.yildirim2@ogr.sakarya.edu.tr  
Sakarya University, Information Systems Engineering
