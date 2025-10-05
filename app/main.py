import streamlit as st
import tensorflow as tf 
from PIL import Image
import numpy as np
import os
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from flask_cors import CORS
import pandas as pd


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/fashion_mnist_model_v2.h5"
# Load the pre-trained model
#model = tf.keras.models.load_model(model_path)
model = tf.keras.models.load_model(model_path, compile=False)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


import json

with open("class_labels.json", "r") as f:
    class_labels = json.load(f)


def get_class_name(index, df):
    try:
        return df.iloc[index]['productDisplayName']
    except IndexError:
        return f"Unknown class index: {index}"




# Function to preprocess the uploaded image (RGB, resized to 96x96)
def preprocess_image(image):
    img = Image.open(image).convert('RGB')  # Ensure 3-channel RGB
    img = img.resize((96, 96))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape((1, 96, 96, 3))  # Model expects 96x96x3
    return img_array


# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')  # Keep it RGB
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))  # Display-friendly size
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)[0]
           
            probabilities = result

            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]* 100
            predicted_label = class_labels[predicted_class]

            st.success(f'Prediction: {predicted_label} ({confidence:.2f}% confidence)')

          



