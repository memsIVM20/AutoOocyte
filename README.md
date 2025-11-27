# 🧫 Self-Driving Microscopy System for Intelligent In-Vitro Imaging of Oocyte Maturation

A fully automated, AI-driven time-lapse microscopy system designed to monitor **oocyte maturation** inside **standard culture dishes** — without requiring customized microwell plates.  
This repository contains the software pipeline for motion control, autofocus, oocyte detection, segmentation, and time-lapse data management as described in the paper:

[> **A self-driving microscopy system for intelligent in vitro imaging of oocyte maturation**](files\Manuscript.docx)

---
![](files\img1.png)

## 📌 Overview

This project introduces a **3-DOF self-driving microscope**, integrated with:

- High-precision X–Y–Z motorized motion platform  
- Tenengrad-based real-time autofocus  
- YOLOv8-based oocyte detection  
- Deep learning segmentation for cumulus expansion  
- Automated time-lapse acquisition for long-term monitoring  

The system achieves:

- **>99% detection accuracy**
- **100% oocyte recovery within wells**
- **Stable 30+ hour continuous imaging**
- Effective imaging of **both oocytes and developing embryos**

It is fully modular and can be extended to stem cells, cancer cells, or drug-response studies.

---

## 🧬 Key Features

![](files\img2.png)

### 🕹 1. Automated Motion Control (3-DOF)
- X-axis: culture dish movement  
- Y, Z axes: camera positioning  
- Step resolution: **0.3125 μm** per microstep  
- Enables precise scanning and autofocus across all wells.

---

### 🔍 2. Intelligent Autofocus (Tenengrad Gradient)
- Robust for low-contrast biological samples  
- Computes sharpness via Sobel gradients  
- Selects optimal Z-plane by maximizing Tenengrad score  
- Performance Metrics:
  - **FWHM**: 5–8 frames  
  - **SNR**: 8.7 → 16.0  
  - **Smoothness Ratio**: 0.54 → 1.42  

---

### 🧠 3. YOLOv8 Object Detection
- Dataset: **4,900+ manually annotated microscope images**  
- Train/Val/Test: 60/20/20  
- Metrics:
  - Precision: **>0.99**
  - Recall: **>0.99**
  - mAP@0.5: **0.934**
- Robust against:
  - Occlusion  
  - Oocyte drifting  
  - Off-center and low-focus samples  

---

### 🎯 4. Cumulus–Oocyte Segmentation
- YOLOv8-seg model for pixel-wise segmentation  
- Quantifies:
  - Oocyte area  
  - Cumulus expansion  
- Enables automated morphological assessment  
- Cumulus growth curves correlate with maturation quality

---

### ⏱ 5. Long-Term Time-Lapse Monitoring
- 30+ hour imaging  
- Full tracking despite oocyte drift  
- Supports embryo development monitoring post-maturation  
- Outputs include:
  - Segmentation masks  
  - Area growth curves  
  - Sharpness profile  
  - Time-coded image stacks  
  - Metadata logs  

---

## 📁 Repository Structure

```
├── backend/
|   ├── camera_utilities/        # Image capture, streaming utilities
│   ├── data/                    # Stored datasets, logging, or exported results
│   ├── detect_main/             # Main detection pipeline (object/egg detection workflow)
│   ├── plc_communication/       # Communication module with PLC (motion control, I/O commands)
│   ├── saved_images/            # Temporary and processed images saved during operation
│   └── segmentation/            # Segmentation models, processing scripts, and post-processing tools
├── files                        # Illustrations and paper
└── README.md
```

---

## 🔬 Example Outputs

- Autofocus
    - Automatically adjusts Z-axis to maximize sharpness of oocyte images in real time.

![](files\img4.png)

- Oocyte Detection
    - Real-time bounding box localization with >99% accuracy.

- Cumulus Segmentation
    - Used to quantify expansion dynamics.

![](files\img3.png)

- Time-Lapse Charts
    - Plots oocyte area vs. time and cumulus expansion.

![](files\img5.png)


--- 

## 📦 Hardware Used
```
Component                     Specification
-------------------           ----------------------------------------
Microscope Camera             5.0 MP color (Shodensha CS500-C)
Microscope Lens               150× magnification
Motion Platform               3-axis stepper motor stage
Culture Dish                  Standard 12-well dish (Ø 1.75 mm)
Chamber Environment           38.5 °C, 5% CO₂, 5–7% O₂
```
---

## 📚 Citation

If you use this system or code, please cite:
A self-driving microscopy system for intelligent in vitro imaging of oocyte maturation.

## 🤝 Acknowledgments

This work was funded by Vingroup Innovation Foundation (VINIF)
Project code: VINIF.2022.DA00030

## 🧩 Future Work
- Predictive modeling using morphokinetic features
- Real-time adaptive feedback for culture optimization
- Support for stem cells, cancer cells, organoids
- Integration with cloud-based dashboards

## 💬 Contact
Your Lab / Your Name
Email: your.email@example.com