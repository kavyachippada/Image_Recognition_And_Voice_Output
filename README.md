# 🧠 Project Title  
**Image-Based Multilingual Text Recognition and Voice Output System**

---

## 📌 What is the project about?

This project aims to **extract text from noisy or low-quality images**, **translate it into different languages**, and then **read it out loud using voice output**. It’s especially useful for:

- 👁️ People with visual impairments  
- 🌍 Multilingual applications (like translating documents)  
- 📄 Digitizing old, low-quality scanned documents  

It combines multiple technologies (image processing, OCR, and text-to-speech) into **one easy-to-use tool** with a simple web interface.

---

## 💡 Why did we build this?

Existing OCR (Optical Character Recognition) systems often fail when the image is noisy (blurred, bad lighting, old scans). Also, they usually only extract text — they don’t translate it or speak it out loud.

So we wanted to create a complete system that:
- Handles **noisy image input**
- **Accurately extracts text**
- **Translates** it into other languages
- **Speaks** it aloud using voice output
- Offers a **simple web interface** anyone can use

---

## 🔧 How does it work?

The system follows these **5 key steps**:

### 1. 📷 Image Denoising  
When a noisy image is uploaded, we apply advanced image cleaning (called denoising).  
Techniques used:
- Morphological operations (to remove noise)
- Adaptive Histogram Equalization (to enhance contrast)
- Median & Gaussian Filtering
- Non-Local Means Denoising (removes fine noise)
- Edge Detection (preserve text structure)

### 2. 🔍 Text Extraction (OCR)  
We use **Tesseract OCR**, an open-source text recognition tool, with multiple configurations:
- Tries different PSM modes (e.g., single blocks, scattered text) to improve accuracy
- Falls back to `pytesseract.image_to_data()` if general OCR fails, to detect words with confidence

### 3. 🌐 Multilingual Translation  
Extracted text is translated into different languages using the **Googletrans library**  
(a free Python wrapper for Google Translate).

### 4. 🔊 Voice Output  
The translated text is converted into speech using **gTTS (Google Text-to-Speech)**.

### 5. 🖥️ User Interface  
Everything is wrapped in a **Streamlit web app**.  
Users can:
- Upload an image  
- Click a button to denoise, extract, translate, and listen  
All with just a few clicks — no technical knowledge needed.

---

## 🧪 Technologies Used

| Category           | Tools/Libraries                                |
|--------------------|-------------------------------------------------|
| Programming        | Python                                          |
| OCR                | Tesseract, pytesseract                          |
| Image Processing   | OpenCV                                          |
| Denoising          | Morphology, Histogram Equalization, Non-Local Means |
| Translation        | Googletrans                                     |
| Text-to-Speech     | gTTS                                            |
| UI Development     | Streamlit                                       |
| Dataset            | Scanned / printed / noisy multilingual images   |

---

## ✅ Key Benefits

- **High Accuracy**: Denoising techniques boost OCR performance even on poor-quality images.  
- **Multilingual**: Supports many languages — useful for global or regional applications.  
- **Voice Enabled**: Helps visually impaired users or those who prefer listening.  
- **Easy to Use**: Clean, simple interface with one-click processing.

---

## 📈 Results & Improvements

- Successfully extracted and spoke out text from noisy images that traditional OCR tools failed on.  
- Works with printed documents, scanned pages, and low-resolution photos.

**Future improvements:**
- Integrate deep learning-based denoising (CNNs or GANs)  
- Support for handwritten text  
- Real-time mobile performance and deployment

---

## 🧑‍💻 Team Members

- **Kavya Chippada**  
- Sri Sai Krishna Chaitanya  
- Chinnakotla Sivananda Kumar  
- Satya Sayanendra Rayudu  

**Supervised by:**  
*Mr. K.P. Sai Rama Krishna*  
Assistant Professor, CSE Dept., SRKR Engineering College

---
