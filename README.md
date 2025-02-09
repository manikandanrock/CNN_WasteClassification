# AICTE Virtual Internship Project

# Waste Classification Using CNN

## 📌 Project Overview

Waste Classification Using CNN is a deep learning-based project that classifies waste into two categories: **Organic** and **Recyclable** using a Convolutional Neural Network (CNN). The model is trained using **TensorFlow/Keras** with **ImageDataGenerator** for preprocessing.

The project includes a **Streamlit app** for user interaction and image classification.

## 📌 Dataset

Download Dataset from this link: [Kaggle - Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data)

## 🏗 Project Architecture

1. **Model Development (Python & TensorFlow/Keras)**

   - Preprocessing waste images using `ImageDataGenerator`
   - Training a CNN model to classify waste
   - Overcoming overfitting issues

2. **Frontend (Streamlit Web App)**

   - **Streamlit UI** to allow users to upload images for classification

3. **Deployment**

   - Streamlit app for quick access

## 🚀 Features

✔️ Classifies waste as **Organic** or **Recyclable**\
✔️ CNN model trained with **TensorFlow/Keras**\
✔️ **Streamlit App** for easy access\
✔️ **Deployed on Hugging Face Spaces**

## 🛠 Technologies Used

- **Python** (TensorFlow, Keras, OpenCV, NumPy, Pandas)
- **Streamlit** (Web App UI)

## 📂 Project Structure

```
Waste-Classification-Using-CNN/
│── Week1/  # Initial implementation and dataset exploration
│── Week2/  # Model development and training
│── Week3/  # Model evaluation and optimization
│── Final_week/  # Finalized project files and deployment setup
    │── app.py  # Streamlit-based UI
    │── requirements.txt  # Requirements
    │── CNN_model/  # Trained CNN model files
│── .gitignore  # Git ignore file
│── README.md  # Project Documentation
```

## 🛠 Setup & Installation

### 🔹 Streamlit App

```bash
cd Final_week
pip install -r requirements.txt
streamlit run app.py
```

## 📌 Deployment Links

- **Streamlit App** : https://waste-classification-mani.streamlit.app/

---

Contributions & feedback are welcome! 😊

