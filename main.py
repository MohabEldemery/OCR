from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract
import re

# Load the EAST text detection model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to perform text detection and recognition
def text_detector(image):
    orig = image
    (H, W) = image.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    expansion_factor = 0.2

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        boundary = 2

        text = orig[startY - boundary:endY + boundary, startX - boundary:endX + boundary]
        text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        textRecognized = pytesseract.image_to_string(text)

        # Remove non-alphanumeric characters using regex
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', textRecognized)

        orig = cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
        orig = cv2.putText(orig, cleaned_text, (endX, endY + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return orig

# Load the input image
image = cv2.imread('image1.jpg')

# Resize the image if needed
image = cv2.resize(image, (640, 320), interpolation=cv2.INTER_AREA)

# Perform text detection and recognition
result = text_detector(image)

# Display the original image and the result
cv2.imshow("Original Image", image)
cv2.imshow("Text Detection and Recognition", result)
cv2.waitKey(1000000)

# Close all windows
cv2.destroyAllWindows()
