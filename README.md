# Classroom Feedback System Documentation


![Demo Preview](media/classfeedback.gif)
## Overview

The Classroom Feedback System is designed to evaluate and analyze student emotions during a class session. It uses a Convolutional Neural Network (CNN) to detect facial expressions in real-time and provides feedback based on the emotional data collected.

## Components

1. **Model Training Script (`train.py`)**
2. **Feedback Generation and Analysis Script (`feedback.py`)**

### Model Training Script (`train.py`)

#### Description

This script trains a Convolutional Neural Network (CNN) to recognize facial expressions from grayscale images. The trained model is then saved for future use in emotion detection.

#### Dependencies

- TensorFlow
- NumPy
- OpenCV
- Keras
- pandas

#### Instructions

1. **Setup Paths**: Update the `train_dir` and `test_dir` variables with the paths to your training and testing data directories.
2. **Run the Script**: Execute `train.py` to train the model.
3. **Model Output**: The trained model will be saved as `facial_expression_model.h5`.

---

### Feedback Generation and Analysis Script (`feedback.py`)

#### Description

This script utilizes the trained CNN model to detect emotions from live video feed. It collects and analyzes data, generating feedback based on the emotional distribution of students.

#### Dependencies

- OpenCV
- TensorFlow
- NumPy
- pandas
- Matplotlib
- Seaborn
- Tkinter

#### Instructions

1. **Prepare the Environment**: Ensure that all required dependencies are installed.
2. **Model Loading**: The script will load the pre-trained model from `facial_expression_model.h5`.
3. **Start Detection**: Click "Start Facial Expression Detection" to begin capturing and analyzing emotions from live video feed.
4. **Analyze Data**: Use "Analyze Excel File" to upload and analyze an Excel file containing emotion data. The analysis results will be displayed as visualizations and feedback.

---

## Conclusion

The Classroom Feedback System combines real-time facial emotion detection with comprehensive data analysis to provide valuable insights into classroom dynamics. By following the above instructions, you can train the model, collect emotional data, and generate actionable feedback to improve classroom experiences.
