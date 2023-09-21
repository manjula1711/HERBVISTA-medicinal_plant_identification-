import streamlit as st
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub

# Load the TensorFlow Hub model
IMAGE_SHAPE = (224, 224)
model = tf.keras.Sequential([hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1', input_shape=IMAGE_SHAPE + (3,))])

# Load labels
with open('labelplants .txt', 'r') as f:
    labels = f.read().splitlines()

st.title('HERBVISTA (Identify your Herbs!)')
st.write(
    """
    <style>
    .stApp {
        # background-color: green;
        background-image:url('https://i.pinimg.com/564x/2a/83/3f/2a833fc3b811928b77551d164e1a12f8.jpg');
        background-repeat: no-repeat;
        background-size: cover;
    }
    .st-title {
        text-align: center;
        font-size: 30px;
        color: #333;
        margin-bottom: 30px;
    }
    .stButton {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 18px;
    }
    .stButton:hover {
        background-color: #45a049;
    }
    .container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
uploaded_image = st.file_uploader("Choose a Herb ...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Here your Herb ..")

    # Preprocess the image
    image = image.resize(IMAGE_SHAPE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    result = model.predict(image)

    if len(labels) > 0:
        predicted_index = np.argmax(result)
        if predicted_index < len(labels) and predicted_index >= 0:
            predicted_label = labels[predicted_index+3]
            st.write(f"Prediction: {predicted_label}")
            
        else:
            st.write("Invalid prediction index")
    else:
        st.write("No labels found. Please check the 'labelplants.txt' file.")
