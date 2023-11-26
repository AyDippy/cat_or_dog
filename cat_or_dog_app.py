import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('cat_or_dog.h5')

# Set the title and description of the Streamlit app
st.title("Cat or Dog Image Classifier")
st.write("Upload an image, and the model will predict whether it's a cat or a dog.")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for the model
    img = image.load_img(uploaded_file, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(img_array)

    # Map the prediction to the class labels
    if prediction[0][0] > 0.5:
        result = "Dog"
    else:
        result = "Cat"

    # Display the result
    st.success(f"Prediction: {result} (Confidence: {prediction[0][0]:.2%})")
