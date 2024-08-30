import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables
model = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
stop_detection = False

def load_model_if_needed():
    global model
    if model is None:
        try:
            model = load_model(r'facial_expression_model.h5')
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}")
            return False
    return True

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

def start_detection():
    global stop_detection
    stop_detection = False

    if not load_model_if_needed():
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Error opening video stream")
        return

    emotion_data = []
    while not stop_detection:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            emotion_label = predict_emotion(face)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            emotion_data.append({'Timestamp': timestamp, 'Emotion': emotion_label})

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Facial Expression Detection', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"emotion_data_{timestamp}.xlsx"
    try:
        df = pd.DataFrame(emotion_data)
        df.to_excel(filename, index=False, engine='openpyxl')
        messagebox.showinfo("Success", f"Excel file saved successfully as {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Error saving to Excel: {e}")

def stop_detection_callback():
    global stop_detection
    stop_detection = True

def analyze_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if file_path:
        analyze_data(file_path)

def analyze_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error loading file: {e}")
        return

    emotion_counts = df['Emotion'].value_counts()
    emotion_percentages = (emotion_counts / emotion_counts.sum()) * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(x=emotion_percentages.index, y=emotion_percentages.values, palette="viridis")
    plt.title("Emotion Distribution in the Classroom")
    plt.xlabel("Emotion")
    plt.ylabel("Percentage (%)")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.pie(emotion_percentages, labels=emotion_percentages.index, autopct='%1.1f%%', colors=sns.color_palette("viridis"))
    plt.title("Emotion Distribution in the Classroom")
    plt.show()

    happy_percentage = emotion_percentages.get('Happy', 0)
    sad_percentage = emotion_percentages.get('Sad', 0)
    fear_percentage = emotion_percentages.get('Fear', 0)
    surprise_percentage = emotion_percentages.get('Surprise', 0)
    disgust_percentage = emotion_percentages.get('Disgust', 0)

    combined_fear_surprise = fear_percentage + surprise_percentage

    feedback = ""
    if happy_percentage == sad_percentage:
        feedback = f"The class was interesting and enjoyable. Happy and sad percentages are equal at {happy_percentage:.2f}%."
    elif sad_percentage > 80:
        feedback = f"The class was boring or intense. Sad percentage is high at {sad_percentage:.2f}%."
    elif combined_fear_surprise > 40:
        feedback = f"The class was not good and stressful. Combined fear and surprise percentages are high at {combined_fear_surprise:.2f}%."
    elif happy_percentage > 70 and sad_percentage < 20:
        feedback = f"The class was positive and engaging with high happiness and low sadness."
    elif sad_percentage > 50 and happy_percentage < 30:
        feedback = f"The class might have been challenging with significant sadness and low happiness."
    elif fear_percentage > 40 and surprise_percentage < 20:
        feedback = f"Students appear anxious or worried with low surprise."
    elif surprise_percentage > 40 and fear_percentage < 20:
        feedback = f"The class had many surprising elements with minimal anxiety."
    elif all(v < 20 for v in emotion_percentages.values()):
        feedback = "The class had a neutral or minimal emotional response overall."
    elif disgust_percentage > 30:
        feedback = f"Students showed signs of discomfort or disgust ({disgust_percentage:.2f}%). It may be worth investigating the causes."

    messagebox.showinfo("Feedback", f"Feedback for the Class: {feedback}")

# Set up the Tkinter GUI window
root = tk.Tk()
root.title("Facial Emotion Recognition")

# Styling
root.configure(bg='#F0F0F0')
title_label = tk.Label(root, text="Facial Emotion Recognition", font=("Helvetica", 16, "bold"), bg='#F0F0F0', fg='#333')
title_label.pack(pady=20)

# Buttons
start_button = tk.Button(root, text="Start Facial Expression Detection", command=start_detection, font=("Helvetica", 12), bg='#4CAF50', fg='white', relief=tk.RAISED)
start_button.pack(pady=10, padx=20, fill=tk.X)

stop_button = tk.Button(root, text="Stop Detection", command=stop_detection_callback, font=("Helvetica", 12), bg='#F44336', fg='white', relief=tk.RAISED)
stop_button.pack(pady=10, padx=20, fill=tk.X)

analyze_button = tk.Button(root, text="Analyze Excel File", command=analyze_file, font=("Helvetica", 12), bg='#2196F3', fg='white', relief=tk.RAISED)
analyze_button.pack(pady=10, padx=20, fill=tk.X)

root.geometry("400x300")
root.mainloop()
