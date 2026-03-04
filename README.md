# WBC Analyzer — AI-Powered White Blood Cell Classification

 BSc Final Project — Sakarya University, Information Systems Engineering, 2026

## About

An end-to-end deep learning system for automated white blood cell classification from peripheral blood smear images. Built with DenseNet121, custom attention mechanisms, and Grad-CAM explainability. Being extended into an Android mobile application.

## Features

- **DenseNet121 + CBAM Attention Block** — Custom architecture for WBC classification
- **Medical Enhanced Filter** — CLAHE + bilateral filtering + selective sharpening
- **Grad-CAM (XAI)** — Visual explanation of model decisions
- **Custom Loss & Activation** — WBC Focal Loss + MedSwish
- **Flask Web Interface** — Drag-and-drop image analysis
- **93% Test Accuracy** on Raabin WBC Dataset (~12,000 images)

## Demo

in progress...

## Model Access

The trained model file (.keras) is not included in this repository due to its size.

📥 Download: [wbc_final_model_densenet.keras](https://drive.google.com/file/d/1NMsJl_3DcdIlfetu5e_AaI-RVHdB2raP/view)

After downloading, place it in:

data/models/

## Quick Start

git clone https://github.com/frissonitte/wbc-analyzer-thesis.git
cd wbc-analyzer-thesis
pip install -r requirements.txt
**READ MODEL ACCESS
python app.py

Roadmap

    DenseNet121 + Attention Block model
    Medical Enhanced preprocessing
    Flask web interface
    Grad-CAM explainability (XAI)
    Android mobile application
    Google Play Store deployment
    TFLite on-device inference


## Previous Version
This project is an extension of the ISE401 design project:
👉 [WBC-Classification-Project](https://github.com/frissonitte/WBC-Classification-Project)


Emirhan Yıldırım — emirhan.yildirim2@ogr.sakarya.edu.tr

Sakarya University, Information Systems Engineering