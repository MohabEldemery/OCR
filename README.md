
https://github.com/user-attachments/assets/3752bb94-f912-4dba-adc2-2411fcdb375e
# 📝 Text Detection and Recognition using OpenCV EAST + Tesseract  

This project implements a **text detection and recognition pipeline** using the **EAST deep learning model** for text detection and **Tesseract OCR** for text recognition.  

The script detects text regions in an image, extracts them, and recognizes text content — then draws bounding boxes and recognized text on the original image.  

---

## ✨ Features  
- 🔍 Detects text regions using **EAST (Efficient and Accurate Scene Text Detector)**  
- 🧠 Recognizes detected text using **Tesseract OCR**  
- 🧹 Cleans text with regex (removes non-alphanumeric characters)  
- 🖼️ Draws bounding boxes + overlays recognized text  
- 🔄 Modular `text_detector()` function for reusability  

---

## ⚙️ Requirements  

- Python **3.8+**  
- **Tesseract OCR** installed  
- Python libraries:  
  - `opencv-python`  
  - `pytesseract`  
  - `imutils`  
  - `numpy`  

### Install dependencies  
```bash
pip install opencv-python pytesseract imutils numpy

Alongside this code, I created a video demonstration (demo.mp4) that walks through how the script works with the code.


https://github.com/user-attachments/assets/38970e59-f18d-4dbd-8f5e-d60a88c6db26

