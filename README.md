# Face Mask Detection using CNN (End-to-End Pipeline)

## Project Overview

This project implements an end-to-end Face Mask Detection system using Convolutional Neural Networks (CNN). The model detects whether a person is:

- Wearing a mask correctly
- Not wearing a mask
- Wearing a mask incorrectly

The project demonstrates the complete Computer Vision pipeline, including data preprocessing, dataset preparation, model training, evaluation, and prediction.

---

## Dataset

**Dataset Source:**
https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

### Classes

- with_mask
- without_mask
- mask_weared_incorrect

### Annotation Format

- Pascal VOC XML
- Bounding Box Annotations

---

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

---

## Project Pipeline

1. Dataset Collection
2. Dataset Splitting
3. XML Annotation Parsing
4. Image Preprocessing
5. CNN Model Development
6. Model Training
7. Model Evaluation
8. Prediction

---

## Folder Structure

```
Face_Mask_Detection_Assignment/
│
├── dataset/
│
├── models/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── xml_parser.py
│
├── app.py
├── split_dataset.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

- CNN-based Face Mask Classification
- Image Preprocessing
- XML Annotation Parsing
- Model Training
- Model Evaluation
- Modular Code Structure
- Easy Deployment

---

## Applications

- Smart Surveillance
- Airports
- Shopping Malls
- Hospitals
- Railway Stations
- Public Places

---

## Future Improvements

- Real-Time Webcam Detection
- Streamlit Web Application
- Mobile Deployment
- Transfer Learning using MobileNet or ResNet
- Cloud Deployment

---

## Installation

Clone the repository

```bash
git clone https://github.com/honey616/face-mask-detection.git
```

Move into the project directory

```bash
cd face-mask-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## Output

The model classifies images into the following categories:

- With Mask
- Without Mask
- Mask Worn Incorrectly

---

## Author

**Honey Upadhyay**

- B.Tech (Computer Science Engineering)
- PG-DAI (CDAC)

GitHub:
https://github.com/honey616/face-mask-detection
