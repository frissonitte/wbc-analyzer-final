# WBC Analyzer — Multi-modal Agentic AI for White Blood Cell Classification
**BSc Final Project — Sakarya University, Information Systems Engineering, 2026**

## About
An end-to-end, multi-modal deep learning system for automated white blood cell classification from peripheral blood smear images. Built with a custom DenseNet121 architecture, CBAM attention mechanisms, and Grad-CAM explainability. 

Taking inspiration from modern Agentic AI workflows, the system integrates an **Autonomous LLM Hematologist** (Gemini 2.5 Flash) that analyzes the CNN's spatial reasoning (heatmaps) to generate real-time clinical reports and detect "Shortcut Learning" phenomena.

## Features
* **DenseNet121 + CBAM Attention Block:** Custom architecture for robust WBC classification.
* **Agentic XAI Reporting (Multi-modal):** Integrates Gemini 2.5 Flash Vision to autonomously interpret Grad-CAM heatmaps and validate model focus.
* **Shortcut Learning Detection:** The LLM agent automatically detects if the model is "cheating" by focusing on the background rather than cellular morphology.
* **Medical Enhanced Filter:** CLAHE + bilateral filtering + selective sharpening preprocessing pipeline.
* **Custom Loss & Activation:** WBC Focal Loss + MedSwish for imbalanced medical datasets.
* **Flask Web Interface:** Real-time drag-and-drop image analysis with interactive UI.
* **93% Test Accuracy** on the Raabin WBC Dataset (~12,000 images).

## Demo
*in progress...*

## Model Access
The trained model file (`.keras`) is not included in this repository due to its size.
📥 **Download:** [wbc_final_model_densenet.keras](https://drive.google.com/file/d/1NMsJl_3DcdIlfetu5e_AaI-RVHdB2raP/view)
After downloading, place it in: `data/models/`

## Quick Start
```bash
git clone [https://github.com/frissonitte/wbc-analyzer-final.git](https://github.com/frissonitte/wbc-analyzer-final.git)
cd wbc-final
pip install -r requirements.txt
# Set your Gemini API Key in your environment variables
export GEMINI_API_KEY="your_api_key_here"
python app.py

Roadmap

    [x] DenseNet121 + Attention Block model

    [x] Medical Enhanced preprocessing

    [x] Flask web interface

    [x] Grad-CAM explainability (XAI)

    [x] Multi-modal LLM Integration (Agentic Reporting)

    [ ] Background Masking / Segmentation (Addressing Shortcut Learning)

    [ ] TFLite on-device inference (Future Work)

    [ ] Android mobile application (Future Work)


## Previous Version
This project is an extension of the ISE401 design project:
👉 [WBC-Classification-Project](https://github.com/frissonitte/WBC-Classification-Project)


Emirhan Yıldırım — emirhan.yildirim2@ogr.sakarya.edu.tr

Sakarya University, Information Systems Engineering
