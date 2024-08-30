import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Function to analyze the data and show results
def analyze_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        messagebox.showerror("Error", f"The file at {file_path} was not found.")
        return
    except Exception as e:
        messagebox.showerror("Error", f"Error loading file: {e}")
        return

    # Calculate the count and percentage of each emotion
    emotion_counts = df['Emotion'].value_counts()
    emotion_percentages = (emotion_counts / emotion_counts.sum()) * 100

    # Visualization: Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=emotion_percentages.index, y=emotion_percentages.values, palette="viridis")
    plt.title("Emotion Distribution in the Classroom")
    plt.xlabel("Emotion")
    plt.ylabel("Percentage (%)")
    plt.show()

    # Visualization: Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(emotion_percentages, labels=emotion_percentages.index, autopct='%1.1f%%', colors=sns.color_palette("viridis"))
    plt.title("Emotion Distribution in the Classroom")
    plt.show()

    # Determine the Majority Emotion (Mode)
    majority_emotion = emotion_counts.idxmax()
    majority_percentage = emotion_percentages[majority_emotion]

    # Provide Feedback Based on the Majority Emotion
    feedback = ""
    if majority_emotion == 'Happy':
        feedback = f"The class is predominantly happy ({majority_percentage:.2f}%). Keep up the positive environment!"
    elif majority_emotion == 'Neutral':
        feedback = f"The class is mostly neutral ({majority_percentage:.2f}%). Consider adding some engaging activities to uplift the mood."
    elif majority_emotion == 'Sad':
        feedback = f"The class seems sad ({majority_percentage:.2f}%). It might be beneficial to introduce some cheerful activities."
    elif majority_emotion == 'Angry':
        feedback = f"The class is showing signs of anger or frustration ({majority_percentage:.2f}%). Consider addressing any potential stress or concerns."
    elif majority_emotion == 'Surprise':
        feedback = f"The class is surprised ({majority_percentage:.2f}%). Ensure everyone is following along and not confused."
    elif majority_emotion == 'Fear':
        feedback = f"The class is fearful ({majority_percentage:.2f}%). Reassurance might be needed to make students feel more comfortable."
    elif majority_emotion == 'Disgust':
        feedback = f"The class is showing signs of discomfort or disgust ({majority_percentage:.2f}%). It may be worth investigating what is causing this reaction."

    messagebox.showinfo("Feedback", f"Feedback for the Class: {feedback}")

# Function to open the file dialog and select a file
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if file_path:
        analyze_data(file_path)

# Set up the GUI window
root = tk.Tk()
root.title("Emotion Analysis")

# Create and place the button
open_button = tk.Button(root, text="Open Excel File", command=open_file)
open_button.pack(pady=20)

# Run the GUI event loop
root.mainloop()
