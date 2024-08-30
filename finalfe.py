import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the model
try:
    model = load_model(r'facial_expression_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def predict_emotion(face_img):
    try:
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = face_img / 255.0  # Normalize
        emotion_probs = model.predict(face_img)
        emotion_label = emotions[np.argmax(emotion_probs)]
        return emotion_label
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return 'Unknown'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video stream")
    exit()

# List to store emotion data
emotion_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        emotion_label = predict_emotion(face)
        
        # Append emotion and timestamp to the list
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        emotion_data.append({'Timestamp': timestamp, 'Emotion': emotion_label})

        # Draw rectangle and put text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Facial Expression Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate a unique filename with a timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
filename = f"C:\\Users\\ASUS\\Downloads\\facialemotionrecognition\\emotion_data_{timestamp}.xlsx"

# Save the collected data to an Excel file
try:
    df = pd.DataFrame(emotion_data)
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"Excel file saved successfully as {filename}")
except Exception as e:
    print(f"Error saving to Excel: {e}")
