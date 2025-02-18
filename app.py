import streamlit as st 
import numpy as np
import tensorflow as tf
from PIL import Image


# Load the trained model
model_path = r"C:\Users\Strange\Documents\data sc\image classifier\models\happysadmodel.keras"
model = tf.keras.models.load_model(model_path)



# Function to preprocess image

def preprocess_image(image):
    image=image.resize((256,256))  # Resize to match model input
    image=np.array(image)/255.0  # Normalize
    if image.shape[-1] == 4:  # If image has an alpha channel (RGBA), remove it
        image = image[:, :, :3]
    image = image.reshape(1, 256, 256, 3)  # Reshape for CNN input (batch_size, height, width, channels)
    return image  


 # Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2d2d2d;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #4b4b4b;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            text-align: center;
        }
        .predict-btn {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        .predict-btn:hover {
            background-color: #45a049;
        }
        .image-container {
            margin: 20px 0;
            padding: 10px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .result-container {
            margin-top: 20px;
            font-size: 24px;
            font-weight: bold;
            color: #2d2d2d;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="title">😊 Happy vs Sad Image Classifier 😢</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let the AI classify it as Happy or Sad.</div>', unsafe_allow_html=True)



 #File uploader handling
uploaded_file = st.file_uploader("Upload an image...", type=['jpeg','jpg','bmp','png'])

# Store the uploaded file in session state to remember it
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file


# If file is uploaded, show the image and prediction
if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file
    image = Image.open(uploaded_file)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess and classify
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)

    # Show prediction result
    label = "Happy 😊" if prediction[0][0] < 0.5 else "Sad 😢"
    st.markdown(f'<div class="result-container">{label}</div>', unsafe_allow_html=True)

