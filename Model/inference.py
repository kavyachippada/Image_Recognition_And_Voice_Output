# Import required libraries
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

# Load the saved model
model = load_model('D:/code/Denoising And Translation/Model/model.h5')

# Load the LabelBinarizer (assuming it was used during training)
# Note: You need to recreate the LB object with the same classes as during training
# For digits 0-9 and A-Z (35 classes), we’ll define it manually
classes = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]  # 0-9, A-Z
LB = LabelBinarizer()
LB.fit(classes)

# Define contour sorting function (same as training)
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

# Define letter extraction function with preprocessing (same as training)
def get_letters(img_path):
    letters = []
    # Read the image
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing: Thresholding and dilation
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    
    # Loop over contours
    for c in cnts:
        if cv2.contourArea(c) > 10:  # Filter small contours
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract ROI and preprocess it
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)  # Resize to 32x32
            thresh = thresh.astype("float32") / 255.0  # Normalize to [0, 1]
            thresh = np.expand_dims(thresh, axis=-1)  # Add channel dimension
            thresh = thresh.reshape(1, 32, 32, 1)  # Reshape to (1, 32, 32, 1)
            
            # Predict
            ypred = model.predict(thresh, verbose=0)
            ypred = LB.inverse_transform(ypred)
            [x] = ypred
            letters.append(x)
    
    return letters, image

# Define word formation function (same as training)
def get_word(letters):
    word = "".join(letters)
    return word

# Test on sample images
sample_images = [
    "D:/code/Denoising And Translation/Model/TRAIN_00003.jpg"
    
]

# Run inference on each sample image
for img_path in sample_images:
    try:
        letters, image = get_letters(img_path)
        word = get_word(letters)
        print(f"Image: {img_path.split('/')[-1]} - Predicted word: {word}")
        plt.imshow(image)
        plt.title(f"Predicted: {word}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
