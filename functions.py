import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from labels import classes  
from PIL import Image

# Function to load the pre-trained model
@st.cache_resource()
def load_trained_model():
    try:
        model = load_model('./model/model.h5')  # Load full model
        return model
    except Exception as e:
        st.error(f'Error loading model: {str(e)}')
        return None

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((30, 30))  # Resize for model
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict label for the uploaded image
def predict_label(image, model):
    if model is None:
        st.error("Model not loaded properly. Check model file.")
        return None

    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        label = classes.get(np.argmax(prediction), "Unknown")  # Handle missing class
        return label
    except Exception as e:
        st.error(f'Error predicting label: {str(e)}')
        return None
