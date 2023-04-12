#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import cv2
import pytesseract
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import model_from_json

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the model architecture from the saved JSON file
with open('model_architecture.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)

# Load the model weights from the saved hdf5 file
model.load_weights('C:/Users/Fujitsu/sgdo-40000r-30e-31136t-3463v.hdf5')

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 

# Define the function to preprocess the image for the second model
def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h-old_h)/2), int((new_h-old_h)/2)+old_h
    w1, w2 = int((new_w-old_w)/2), int((new_w-old_w)/2)+old_w
    img_pad = np.ones([new_h, new_w]) * 255
    img_pad[h1:h2, w1:w2] = img
    return img_pad

def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w<target_w and h<target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w>=target_w and h<target_h:
        new_w = target_w
        new_h = int(h*new_w/w)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w<target_w and h>=target_h:
        new_h = target_h
        new_w = int(w*new_h/h)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        
        ratio = max(w/target_w, h/target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img

def process_image(img):
    """ Pre-processing image for predicting """
 
    img = fix_size(img, 128, 32)

    #img = np.clip(img, 0, 255)
    #img = np.uint8(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.subtract(255, img)
    
    img = np.expand_dims(img, axis=2)
    
    img = img.astype(np.float32)
    img /= 255
    return img

# Define the first Streamlit app section for the first model
def app1():
    st.title("Image to Text Converter")
    

    st.write("Upload an image containing text and the app will predict the text.")    

    # Define image upload widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Define language selection widget
    language = st.selectbox("Select language", ["English", "German"])

    # Define text prediction function
    def predict_text(image, language):
        """
        page segmentation mode (psm):
      
        0    Orientation and script detection (OSD) only.
        1    Automatic page segmentation with OSD.
        2    Automatic page segmentation, but no OSD, or OCR.
        3    Fully automatic page segmentation, but no OSD. (Default)
        4    Assume a single column of text of variable sizes.
        5    Assume a single uniform block of vertically aligned text.
        6    Assume a single uniform block of text.
        7    Treat the image as a single text line.
        8    Treat the image as a single word.
        9    Treat the image as a single word in a circle.
        10    Treat the image as a single character.
        11    Sparse text. Find as much text as possible in no particular order.
        12    Sparse text with OSD.
        13    Raw line. Treat the image as a single text line,bypassing hacks that are Tesseract-specific.
     
        OCR Engine Mode (oem):
      
        0    Legacy engine only.
        1    Neural nets LSTM engine only.
        2    Legacy + LSTM engines.
         3    Default, based on what is available.
        """
        # Load the appropriate language model for Tesseract
        if language == "English":
            config = "-l eng --oem 3 --psm 3"    #page
        else:
            config = "-l deu --oem 3 --psm 3"
                
        # Convert image to grayscale and apply thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply OCR using Tesseract
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip()
        
        
    # Predict text when image is uploaded
    if uploaded_file is not None:
        # Read image bytes and convert to OpenCV image format
        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Predicted Text:")
        text = predict_text(image, language)
        st.write(text)



# Define the second Streamlit app section for the second model
def app2():
    st.title("Image to Text Converter")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 0)

        # Preprocess the image
        img = process_image(img)

        # Reshape the image to fit the model input shape
        img = np.reshape(img, (1, 32, 128, 1))

        # Make the prediction using the loaded model
        prediction = model.predict(img)
        decoded = K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0]
        out = K.get_value(decoded)

        # Display the predicted text
        st.subheader("Prediction:")
        predicted_text = ""
        for p in out[0]:
            if int(p) != -1:
                predicted_text += char_list[int(p)]
        st.write(predicted_text)

        # Display the uploaded image
        st.subheader("Uploaded Image:")
        st.image(uploaded_file, use_column_width=True)

# Set page title and favicon
st.set_page_config(page_title="TextTech", page_icon=":pencil2:")

# Define the navigation menu
menu = ["Word Text Recognition Model", "Image to Text Converter"]
choice = st.sidebar.selectbox("Select a model", menu)

# Set page width and height
st.markdown("<style>.reportview-container{max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem;}</style>", unsafe_allow_html=True)

if choice == "Word Text Recognition Model":
    app2()
else:
    app1()

# Add footer text
st.markdown("<div style='text-align: center; padding-top: 1rem;'>Created by Zishan and Kalpana</div>", unsafe_allow_html=True)

