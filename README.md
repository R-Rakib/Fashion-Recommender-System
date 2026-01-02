# Fashion-Recommender-System
dataset:https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

# 👗 Fashion Recommender System using ResNet50

A **content-based fashion recommendation system** that suggests visually similar clothing items using **deep learning**.  
The system leverages **ResNet50** for feature extraction and **Streamlit** for an interactive web-based interface.

---

## 📌 Overview

This project uses a **pre-trained ResNet50 model** to extract deep visual features from fashion images.  
Based on these features, the system recommends similar fashion items when a user uploads an image.

The goal is to demonstrate how **Computer Vision + Deep Learning** can be applied to real-world recommendation systems without relying on textual data.

---

## 🧠 How It Works

1. User uploads an image through the **Streamlit web app**
2. ResNet50 extracts high-level visual features
3. Feature vectors are compared using similarity measures
4. Top visually similar fashion items are recommended

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Deep Learning:** ResNet50 (Pre-trained CNN)  
- **Frameworks & Libraries:**  
  - TensorFlow / Keras  
  - NumPy  
  - Scikit-learn  
- **Web Framework:** Streamlit  
- **Computer Vision:** Image preprocessing & feature extraction  


│── requirements.txt
│── README.md
