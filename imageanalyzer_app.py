import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Loading saved model
@st.cache_resource
def load_cnn_model():
    return load_model('imageanalyzer_cnn_model.h5')

model = load_cnn_model()

# Preprocessing the loaded image
def preprocess_image(image):
    # Converting to grayscale
    image = image.convert("L")
    # Resizing the image
    image = image.resize((28, 28))
    # Convert to array
    image = img_to_array(image)
    # Normalizing the pixel values
    image = image / 255.0
    # Adding batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload a handwritten digit image, and the model will predict the digit.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Displaying the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocessing the image
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)

    # Predicting the digit
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)

    # Displaying the prediction
    st.write(f"Predicted Digit: {predicted_digit}")
