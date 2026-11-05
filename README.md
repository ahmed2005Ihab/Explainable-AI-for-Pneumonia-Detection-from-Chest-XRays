# Explainable AI for Pneumonia Detection from Chest X-Rays

A comparative study of **nine deep learning architectures** and **ten explainability (XAI) methods** for detecting pneumonia on the RSNA Pneumonia Detection Challenge dataset.

Course project for **DSAI 305: Explainable AI**, Zewail City of Science and Technology (Team 19, May 2026).

**Team:** Ahmed Ihab, Ahmed Assem, Habiba Ayman

**Repository:** [github.com/ahmed2005Ihab/Explainable-AI-for-Pneumonia-Detection-from-Chest-XRays](https://github.com/ahmed2005Ihab/Explainable-AI-for-Pneumonia-Detection-from-Chest-XRays)

## Overview

Deep learning models can classify chest X-rays well, but clinicians cannot act on predictions they cannot inspect. This project trains several architectures on the same task and then asks whether their explanations point at the lung regions a radiologist marked, or at something else.

The study addresses three questions:

1. How do the nine architectures compare in accuracy, AUC, and F1 on binary pneumonia detection?
2. Do XAI explanations show that models focus on clinically meaningful lung regions, or do they reveal spurious cues?
3. Which model and XAI combination gives the best balance of performance, spatial faithfulness, and computational cost?

## Dataset

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge): 26,684 patients with frontal chest X-rays in DICOM format (1024x1024).

| Class | Patients |
| --- | --- |
| Normal | 8,851 |
| No Lung Opacity / Not Normal | 11,821 |
| Lung Opacity (pneumonia) | 6,012 |

Radiologist-annotated bounding boxes for the positive cases are used as ground truth when evaluating the explanations. Each model group used its own split, class balancing, and preprocessing (for example CLAHE contrast enhancement, per-image Z-score normalization, and resizing to the model's native resolution). The details are in the paper.

## Models

| Model | Input size | Notes |
| --- | --- | --- |
| COVID-Net (PEPX) | 480x480 | Adapted to binary classification, class weights |
| Vision Transformer (COVID-Transformer) | 224x224 | Frozen ViT L-16 backbone with a custom MLP head |
| VGG-based CNN | 224x224 (grayscale) | Lightweight CNN, balanced subset |
| CheXNet (DenseNet-121) | 224x224 | Pretrained backbone, positive-class weighting |
| EfficientNet-B4 | 380x380 | Pretrained backbone, balanced subset |
| Xception | 299x299 | Focal loss |
| CNN-XGBoost hybrid | 64x64 | CNN features (128-d) fed to an XGBoost classifier |
| MobileNetV2 | 224x224 | Trained from scratch, 5-fold cross-validation |
| CNN-YOLOv11 | 64x64 | Custom CNN with an Ultralytics YOLOv11 pipeline |

## XAI methods

| Method | Applied to |
| --- | --- |
| Grad-CAM | All 9 models |
| SHAP | All 9 models |
| LIME | 8 models (all except the VGG-based CNN) |
| Integrated Gradients | COVID-Net, CheXNet, EfficientNet-B4, Xception |
| Guided Grad-CAM | CNN-XGBoost, CNN-YOLOv11 |
| Saliency Maps | VGG-based CNN |
| Occlusion Sensitivity | VGG-based CNN, MobileNetV2 |
| RISE, Blur Ablation, Spotlight (inverse occlusion) | Vision Transformer |

**Quantitative XAI evaluation.** For CheXNet and EfficientNet-B4, Grad-CAM was compared with the radiologist boxes using two metrics:

- **Pointing Game:** the share of positive patients whose Grad-CAM peak falls inside the radiologist's box.
- **CAM Intensity Ratio:** mean activation inside the box divided by mean activation outside it (above 1 means the model focuses on the lesion).

## Results

| Model | Accuracy | Pneumonia precision | Pneumonia recall | Macro F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| COVID-Net (PEPX) | 0.80 | 0.55 | 0.71 | 0.75 | 0.876 |
| ViT (COVID-Transformer) | 0.77 | 0.49 | 0.78 | 0.72 | n/a |
| VGG-based CNN | 0.77 | 0.73 | 0.84 | 0.77 | 0.854 |
| CheXNet (DenseNet-121) | 0.79 | 0.69 | 0.85 | 0.79 | 0.8804 |
| EfficientNet-B4 | 0.81 | 0.71 | 0.85 | 0.80 | 0.8948 |
| Xception | 0.82 | 0.74 | 0.80 | 0.81 | 0.8903 |
| CNN-XGBoost | 0.8011 | 0.80 | 0.80 | 0.80 | n/a |
| MobileNetV2 | 0.7686 | 0.78 | 0.77 | 0.77 | n/a |
| CNN-YOLOv11 | 0.7756 | 0.78 | 0.78 | 0.78 | n/a |

CheXNet, EfficientNet-B4, and Xception are reported at their F1-optimal thresholds (0.3975, 0.4243, and 0.5093).

**Explanation quality (10 positive patients each):**

| Model | Pointing Game | Mean CAM Intensity Ratio |
| --- | --- | --- |
| CheXNet | 40% (4/10) | 1.66 |
| EfficientNet-B4 | 30% (3/10) | not reported |

## Key findings

- **Best scores:** Xception had the highest accuracy (0.82) and macro F1 (0.81), and EfficientNet-B4 had the highest AUC (0.8948).
- **Threshold tuning matters.** Lowering CheXNet's threshold from 0.50 to 0.3975 raised pneumonia recall from 0.79 to 0.85 and cut false negatives from 193 to 131.
- **Accuracy and explanation quality do not always agree.** EfficientNet-B4 had the highest AUC but a lower Pointing Game score than CheXNet. LIME still highlighted the correct opacity regions for it, which suggests the model learned useful features and the problem is in the Grad-CAM pathway. The paper attributes this to the architecture's aggressive downsampling.
- **Methods broadly agree on true positives.** Grad-CAM, LIME, SHAP, and Integrated Gradients mostly highlighted the lower lung zones.
- **False positives are informative.** XAI maps suggested attention to the cardiac silhouette and ribs (VGG-based CNN) and to high-frequency image artifacts (ViT).
- **Use more than one XAI method.** Gradient-based and perturbation-based explanations gave complementary views.

## Limitations

- Each model group used its own data split, balancing strategy, and test set (about 700 to 4,000 patients), so the table is not a strict head-to-head comparison.
- The Pointing Game and CAM Intensity Ratio were computed on only 10 positive patients per model.
- The ViT backbone was kept frozen, and no AUC is reported for it.
- This is a research study on a public dataset and is not clinically validated.

## Repository contents

| File | Description |
| --- | --- |
| `DSAI_305_XAI_Final_Research_Paper_Team_19.pdf` | Full 13-page research paper |
| `XAI_Pneumonia_Detection_Team19.pptx` | Presentation slides |

## Acknowledgements

Dataset: Radiological Society of North America (RSNA) Pneumonia Detection Challenge, hosted on Kaggle.
