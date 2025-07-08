import streamlit as st
from streamlit_option_menu import option_menu
from googletrans import Translator
import time
from PIL import Image
import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from img_denoise import denoise
from ocr_text import read_text
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import imutils
from gtts import gTTS
import io
import google.generativeai as genai

translator = Translator()

language_options = {
    "Telugu": "te",
    "Tamil": "ta",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru"
}

st.markdown("""
    <style>
    .main {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .title {font-family: 'Arial', sans-serif; color: #1f77b4; text-align: center; font-size: 36px; font-weight: bold; margin-bottom: 20px;}
    .subheader {font-family: 'Arial', sans-serif; color: #ff7f0e; font-size: 24px; margin-top: 20px;}
    .translated-box {background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 15px; font-size: 18px; color: #2c3e50;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_handwriting_model():
    model_path = 'C:/Project/Model/model.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None, None
    model = load_model(model_path)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    LB = LabelBinarizer()
    LB.fit(classes)
    return model, LB

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top": reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top": i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def get_letters(image, model, LB):
    letters = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1, 32, 32, 1)
            ypred = model.predict(thresh, verbose=0)
            ypred = LB.inverse_transform(ypred)
            [x] = ypred
            letters.append(x)
    return letters, image

def get_word(letters):
    return "".join(letters)

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        st.audio(audio_file, format="audio/mp3")
    except Exception as e:
        st.error(f"TTS failed: {str(e)}")

def translate_page():
    st.markdown('<div class="title">🌍 Multilingual Text Translator</div>', unsafe_allow_html=True)
    
    # Initialize session state for translation
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = None

    img = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="translate_uploader")
    if img is None:
        st.warning("Please upload an image.")
    else:
        img = denoise(img)
        user_input = read_text(img)
        st.write("Extracted Text:", user_input)

        st.write("### Select Language")
        selected_language = st.selectbox("Choose a language:", list(language_options.keys()), key="language_dropdown")
        
        if st.button("Translate Now", key="translate_button") and user_input.strip():
            with st.spinner("Translating..."):
                time.sleep(1)
                try:
                    translated_text = translator.translate(user_input, dest=language_options[selected_language]).text
                    st.session_state.translated_text = translated_text
                    st.session_state.selected_language = selected_language
                    
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")

        # Display translated text if available
        if st.session_state.translated_text and st.session_state.selected_language:
            st.markdown('<div class="subheader">Translated Text</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="translated-box"><b>{st.session_state.selected_language}:</b> {st.session_state.translated_text}</div>',
                unsafe_allow_html=True
            )
            if st.button("🔊 Speak", key=f"speak_{st.session_state.selected_language}"):
                text_to_speech(st.session_state.translated_text, language_options[st.session_state.selected_language])

        

def handwriting_page():
    st.markdown('<div class="title">✍️ Handwriting Recognition</div>', unsafe_allow_html=True)
    model, LB = load_handwriting_model()
    if model is None:
        return
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="handwriting_uploader")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        with st.spinner("Processing..."):
            try:
                letters, processed_image = get_letters(image_color, model, LB)
                word = get_word(letters)
                st.write(f"Predicted Word: **{word}**")
                st.image(processed_image, channels="BGR", caption="Processed Image")
                st.write("Raw predicted letters:", letters)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

def threed_text_page():
    st.markdown('<div class="title">📐 3D Text Recognition</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a 3D Text Image", type=["png", "jpg", "jpeg"], key="3d_uploader")
    if uploaded_file is not None:
        genai.configure(api_key="AIzaSyB4VC7om1ZeTznNmSN-LjI6mFiNCYBqSQ0")
        model = genai.GenerativeModel("gemini-1.5-pro")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded 3D Text Image")
        with st.spinner("Processing"):
            try:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                response = model.generate_content([
                    "Extract the text from this 3D text image.", 
                    {"mime_type": "image/png", "data": img_bytes}
                ])
                st.write("Extracted Text:", response.text)
            except Exception as e:
                st.error(f"Error processing 3D text: {str(e)}")

def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Translate", "Handwriting Recognition", "3D Text Recognition"],
            icons=["globe", "pencil", "cube"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#f0f2f6"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
    if selected == "Translate":
        translate_page()
    elif selected == "Handwriting Recognition":
        handwriting_page()
    elif selected == "3D Text Recognition":
        threed_text_page()

if __name__ == "__main__":
    main()