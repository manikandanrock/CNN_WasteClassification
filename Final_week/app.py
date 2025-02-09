import os
import io
import base64
import logging
import numpy as np
import tensorflow as tf
import gdown
from PIL import Image
import streamlit as st

# Initialize logging
logging.basicConfig(level=logging.INFO)

MODEL_ID = "1FMvhvLE2ikEmeIgNFMM8jlnpI_iRSTIn"  # Replace with your actual model ID
MODEL_PATH = "./model.tflite"
CLASS_NAMES = ['Organic', 'Recycleable']  # Replace with your class names

# Function to download the model
def download_model():
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Model file '{MODEL_PATH}' not found. Downloading...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
            file_size = os.path.getsize(MODEL_PATH)
            logging.info(f"Model downloaded. Size: {file_size} bytes")
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            raise
    else:
        logging.info(f"Model file '{MODEL_PATH}' already exists. Skipping download.")

download_model()

# Initialize TFLite interpreter
interpreter = None
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading TFLite model: {e}")
    raise

# Image preprocessing function
def preprocess_image(file):
    try:
        img_bytes = io.BytesIO(file.read())
        img = Image.open(img_bytes).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0).astype(np.float32)
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise

# Streamlit UI Styling
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: 700;
            color: #4CAF50;
            text-align: center;
            margin-top: 20px;
        }
        .subtitle {
            font-size: 18px;
            color: #666;
            text-align: center;
            margin-top: 10px;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            color: #FF5722;
            text-align: center;
            margin-top: 20px;
        }
        .confidence {
            font-size: 20px;
            font-weight: 600;
            color: #2196F3;
            text-align: center;
            margin-top: 5px;
        }
        .file-uploader {
            display: block;
            margin: 20px auto;
            padding: 12px 20px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            border: none;
        }
        .file-uploader:hover {
            background-color: #45a049;
        }
        .footer {
            font-size: 16px;
            color: #888;
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
        .footer p {
            margin: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Title and instructions
st.markdown("<div class='title'>Waste Classification Using CNN</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image to classify it as either 'Organic' or 'Recycleable'.</div>", unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# Button for Prediction
if uploaded_file is not None:
    # Preprocess the image and predict
    img_array = preprocess_image(uploaded_file)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class and score
    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_class_index]
    predicted_score = prediction[0][predicted_class_index] * 100  # Convert to percentage

    # Create layout for displaying results and image
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    with col2:
        st.markdown(f"<div class='prediction'>Prediction: {predicted_class}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence'>Confidence: {predicted_score:.2f}%</div>", unsafe_allow_html=True)

    # Optional: Show the processed image as base64 for sharing
    img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

# Footer Content with styling
st.markdown("""
    <div class="footer">
        <p><strong>MY AICTE INTERNSHIP PROJECT</strong><br>
        This project uses a CNN model to classify waste into Organic or Recyclable.</p>
        <p>Developed by Manikandan G</p>
        <p>Thanks for visiting!</p>
    </div>
""", unsafe_allow_html=True)
