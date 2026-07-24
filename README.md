# ThyroGuide: Explainable AI Assistant for Thyroid Ultrasound Diagnosis

> **AI-powered clinical assistant for thyroid nodule detection, malignancy classification, explainability, and natural-language reporting using Gemma 4.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem](#problem)
- [Solution](#solution)
- [Key Features](#key-features)
- [How Gemma 4 is Used](#how-gemma-4-is-used)
- [System Workflow](#system-workflow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Notebook](#notebook)
- [Demo Video](#demo-video)
- [Clinical Value & Impact](#clinical-value--impact)
- [Future Improvements](#future-improvements)
- [Limitations & Responsible Use](#limitations--responsible-use)
- [Getting Started](#getting-started)
- [Hackathon Highlights](#hackathon-highlights)
- [References](#references)
- [License](#license)
- [Contributors](#contributors)

---

## Project Overview
**ThyroGuide AI** is an explainable, end-to-end clinical decision-support assistant for thyroid ultrasound interpretation.  
It combines computer vision and LLM reasoning to transform model outputs into **clinically interpretable**, **patient-friendly** reports.

The pipeline:
1. Detects nodules in ultrasound images.
2. Localizes ROI.
3. Classifies nodules as **Benign** or **Malignant**.
4. Generates Grad-CAM explainability.
5. Uses **Gemma 4** to produce structured clinical narrative output.

---

## Problem
Thyroid Cancer is the fifth most common cancer in the USA today.[1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8986939/)  
Its incidence and mortality are expected to increase by **29.9%** and **67%**, respectively, by 2040.[1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8986939/)

Women are affected significantly more often than men. Despite growing awareness, ultrasound interpretation remains challenging:
- Some nodules are too small for reliable assessment.
- Small cancers and follicular-pattern tumors can mimic benign lesions.
- Overlapping imaging features increase ambiguity.

As a result, missed or delayed malignant nodule identification can postpone treatment and worsen outcomes.

While existing AI systems estimate ACR TI-RADS scores and classify nodules, many still lack:
- Integrated explainability,
- Automatic ROI localization,
- Clinically interpretable AI-generated reporting.

---

## Solution
**ThyroGuide AI** is an AI-powered Clinical Assistant built with **Gemma 4** that converts thyroid nodule detection and classification outputs into clinically interpretable reports.

Users upload thyroid ultrasound images.  
The system then automatically:

- ✔️ Detects thyroid nodules using **YOLO11m**  
- ✔️ Localizes and extracts **Region of Interest (ROI)** from detected areas  
- ✔️ Classifies nodules as **Benign** or **Malignant** using **CNN**  
- ✔️ Provides **Grad-CAM** explainability for classification validity  
- ✔️ Uses **Gemma 4** to generate a structured, patient-friendly, clinically interpretable report explaining findings, confidence, and limitations

Instead of being a raw classifier, ThyroGuide functions as an **intelligent clinical assistant** bridging AI predictions and human understanding—supporting transparent clinician-patient communication.

---

## Key Features
- **End-to-end pipeline**: Detection → ROI extraction → Classification → Explainability → Report generation  
- **Explainable AI (XAI)**: Grad-CAM heatmaps to support trust and interpretability  
- **Clinical communication layer**: Gemma 4 converts technical outputs into natural-language insights  
- **Decision-support focused**: Designed to augment, not replace, clinical judgment  
- **Patient-centered reporting**: Improves understanding and shared decision-making

---

## How Gemma 4 is Used
Gemma 4 is the **reasoning and report-generation engine** behind ThyroGuide AI.

It:
- Interprets structured outputs from YOLO11m, CNN, and Grad-CAM modules  
- Synthesizes detection results, classification predictions, confidence scores, and explainability cues  
- Produces structured reports for both clinicians and patients  
- Explicitly communicates model confidence and limitations for transparency  
- Bridges algorithmic outputs and clinically meaningful interpretation

> Gemma 4 does **not** perform detection/classification directly; it is the intelligent interpretation layer.

---

## System Workflow
![ThyroGuide AI Workflow](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F24769518%2F09cda92fa23b1875b3883362e5e4138d%2FThyroGuide_AI_Workflow.drawio.png?generation=1784824000838587&alt=media)

**Figure:** Established workflow of ThyroGuide AI.

---

## Tech Stack
- **Detection:** YOLO11m  
- **Classification:** Convolutional Neural Network (CNN)  
- **Explainability:** Grad-CAM  
- **Reasoning & Report Generation:** Gemma 4  
- **Development Environment:** Jupyter/Colab Notebook workflow

---

## Project Structure
```text
ThyroGuide-Explainable-AI-Assistant-for-Thyroid-Ultrasound-Diagnosis/
├── README.md
├── notebooks/
│   └── ThyroGuide_Gemma4.ipynb
├── models/
│   ├── yolo11m_weights/
│   └── cnn_classifier/
├── outputs/
│   ├── detections/
│   ├── roi/
│   ├── gradcam/
│   └── reports/
└── assets/
    ├── workflow.png
    └── demo-video-link.txt
```
> Update paths as needed to match your actual repository files.

---

## Notebook
The project notebook is included and demonstrates:
- End-to-end model flow,
- Integration with Gemma 4,
- Explainability outputs,
- Structured report generation process.

---

## Demo Video
A demo video is attached/provided in the project resources and showcases:
- Image upload,
- Nodule detection and ROI localization,
- Benign/Malignant classification,
- Grad-CAM visualization,
- Final Gemma 4-generated clinical report.

---

## Clinical Value & Impact
ThyroGuide is designed to:
- Improve consistency in preliminary AI-assisted interpretation,
- Reduce black-box perception through visual + textual explainability,
- Save clinician review time by summarizing technical outputs,
- Enhance patient understanding with human-readable reporting.

---

## Future Improvements
1. Expand training with larger expert-annotated real-time thyroid ultrasound datasets.  
2. Improve full-pipeline accuracy and optimize inference speed for near real-time clinical usage.  
3. Add multilingual report generation via Gemma 4 for broader accessibility.  
4. Build an interactive AI assistant for clinician/patient Q&A with evidence-based educational support and clear non-diagnostic boundaries.

---

## Limitations & Responsible Use
- This project is a **clinical decision-support prototype**, not a standalone diagnostic system.  
- Outputs should always be reviewed by qualified healthcare professionals.  
- Performance may vary across populations, devices, and imaging protocols.  
- Explainability improves transparency but does not guarantee correctness.

---

## Getting Started
1. Clone the repository  
   ```bash
   git clone https://github.com/anikasadiaOPT/ThyroGuide-Explainable-AI-Assistant-for-Thyroid-Ultrasound-Diagnosis.git
   cd ThyroGuide-Explainable-AI-Assistant-for-Thyroid-Ultrasound-Diagnosis
   ```

2. Open the notebook in Jupyter/Colab.

3. Install dependencies (customize as needed):
   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook cells step by step to execute the full pipeline.

---

## Hackathon Highlights
- ✅ Real-world healthcare problem with measurable relevance  
- ✅ End-to-end integrated AI pipeline  
- ✅ Explainable AI + LLM reasoning for trust and usability  
- ✅ Clinician- and patient-oriented output design  
- ✅ Strong potential for translation into practical clinical workflows

---

## References
1. National Library of Medicine (PMC):  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC8986939/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8986939/)
2. TN5000: An Ultrasound Image Dataset for Thyroid Nodule Detection and Classification
   [https://figshare.com/s/cb6a67f17c04b29e7edd](https://figshare.com/s/cb6a67f17c04b29e7edd)
3. TN5000: An Ultrasound Image Dataset for Thyroid Nodule Detection and Classification(Paper)
   [https://www.nature.com/articles/s41597-025-05757-4](https://www.nature.com/articles/s41597-025-05757-4)

---

## License
Specify your project license here (e.g., MIT, Apache-2.0, or custom academic/research use license).

---

## Contributors
- **Anika Sadia**  
  GitHub: [@anikasadiaOPT](https://github.com/anikasadiaOPT)

---

> If you use this work in research, demos, or derivative systems, please provide appropriate attribution.
