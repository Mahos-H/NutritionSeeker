# NutritionSeeker

*Author:* Soham Haldar

**Live Demo:** [NutritionSeeker Streamlit App](https://nutritionseeker-jqq8idjmjseszjzuzahdlc.streamlit.app/)  

NutritionSeeker is an advanced multi-modal deep learning system designed to assist dietitians and nutrition experts by estimating nutritional values from food images. By combining state-of-the-art computer vision and natural language processing techniques, NutritionSeeker robustly processes diverse food imagery and predicts key nutritional attributes. The system integrates:

- *YOLOv8* for real-time food object detection.
- *SAM2 (Segment Anything Model 2)* for precise segmentation.
- *ResNet18* for visual feature extraction.
- *BERT* for encoding semantic textual cues derived from detected food labels.

These components are fused into a unified representation that is passed through specialized regression heads tailored to three distinct nutritional datasets:
- *Food Nutrition (e.g., homemade foods)*
- *Fruits & Vegetables*
- *Fast Food*

A user-friendly Streamlit web application is provided for real-time inference, making NutritionSeeker an end-to-end solution for nutritional analysis.

---

## Table of Contents

- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
  - [Input & Preprocessing](#input--preprocessing)
  - [Object Detection with YOLOv8](#object-detection-with-yolov8)
  - [Segmentation with SAM2](#segmentation-with-sam2)
  - [Visual Feature Extraction (ResNet18)](#visual-feature-extraction-resnet18)
  - [Textual Feature Extraction (BERT)](#textual-feature-extraction-bert)
  - [Feature Fusion and Regression Heads](#feature-fusion-and-regression-heads)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Training Procedure](#training-procedure)
- [Streamlit Web Application](#streamlit-web-application)
- [Usage Instructions](#usage-instructions)
- [Additional Technical Details](#additional-technical-details)
- [License](#license)

---

## Overview

NutritionSeeker addresses the challenge of estimating nutritional values from heterogeneous food images. The system isolates food items from complex backgrounds and leverages both visual and semantic cues to generate precise nutritional predictions. This multi-modal approach improves robustness and accuracy compared to traditional single-stream methods, making it an ideal tool for dietitians and nutritionists.

---

## Technical Architecture

The NutritionSeeker pipeline is composed of several detailed stages:

### Input & Preprocessing

- *Image Acquisition:*  
  - Supports image input via file upload or webcam capture (JPG, JPEG, PNG).
- *Preprocessing:*  
  - Converts images to RGB format.
  - Resizes images as needed for subsequent processing modules.

### Object Detection with YOLOv8

- *Model:*  
  - Uses YOLOv8 (e.g., yolov8n.pt) for fast and accurate detection of food objects.
- *Output:*  
  - Returns bounding boxes and a predicted class label (e.g., "burger", "salad").
- *Technical Note:*  
  - The YOLOv8 model is set to evaluation mode; its outputs define the Region of Interest (ROI) for further processing.

### Segmentation with SAM2

- *Model:*  
  - SAM2 refines the ROI by generating segmentation masks.
- *Output:*  
  - Produces a segmented image with the background suppressed, highlighting only the food item.
- *Technical Note:*  
  - Multiple candidate masks are generated; the mask with the largest area is selected.

### Visual Feature Extraction (ResNet18)

- *Model:*  
  - A pretrained ResNet18 model (with the final fully connected layer removed) extracts visual features.
- *Preprocessing:*  
  - The segmented image is resized to 224×224.
  - Normalized using ImageNet statistics and converted to a tensor.
- *Output:*  
  - Extracts a 512-dimensional feature vector.

### Textual Feature Extraction (BERT)

- *Input Construction:*  
  - A text prompt is created from the predicted class label (e.g., "This is a burger food item.").
- *Model:*  
  - A pretrained BERT model (bert-base-uncased) encodes the text prompt.
- *Output:*  
  - Uses the [CLS] token to produce a 768-dimensional text feature vector.
- *Technical Note:*  
  - BERT parameters can be optionally frozen to simplify training.

### Feature Fusion and Regression Heads

- *Fusion:*  
  - The 512-dimensional visual features and 768-dimensional text features are concatenated into a 1280-dimensional vector.
- *Dimensionality Reduction:*  
  - A fully connected layer reduces the fused vector to 512 dimensions.
- *Regression Heads:*  
  - Three separate regression heads output nutritional predictions:
    - *Food Nutrition Head:* (e.g., 3-dimensional output for homemade foods)
    - *Fruits & Vegetables Head:* (e.g., 9-dimensional output)
    - *Fast Food Head:* (e.g., 8-dimensional output)
- *Loss Function:*  
  - Mean Squared Error (MSE) loss is used to train the regression outputs.

---

## Project Structure
