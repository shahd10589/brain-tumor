
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('هنا.h5')

def preprocess_image(image):
    '''Preprocess the uploaded image for the model'''
    image = image.resize((224, 224))  
    image = np.array(image)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0) 
    return image

st.title('Brain Tumor Detection')

uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)

    processed_image = preprocess_image(image)
    st.write("Classifying...")
    prediction = model.predict(processed_image)

    if prediction > 0.5:
        st.write("Prediction: Brain Tumor Detected")
    else:
        st.write("Prediction: No Brain Tumor Detected")

