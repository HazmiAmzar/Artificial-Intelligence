# Music Recommendation System Based on Facial Expression Using Deep Learning

## Project Overview
This project presents a music recommendation system that suggests music tracks based on the user's facial expression. The system uses a Convolutional Neural Network (CNN) to classify the user's emotion from their facial expression and then recommends music corresponding to the recognized emotion. The supported emotions are happy, surprise, sad, anger, disgust, fear, and neutral. The system can operate in both real-time (via webcam) and offline (via static images).

## Key Features
- **Emotion Classification**: Classifies facial expressions into one of seven emotions using deep learning (CNN).
- **Music Recommendation**: Recommends music based on the recognized emotion.
- **Real-Time and Offline Methods**: Supports both real-time face detection using a webcam and offline analysis from images.
- **User-Friendly Interface**: Provides an easy-to-use graphical interface for interaction.

## Technologies Used
- **Deep Learning**: Convolutional Neural Network (CNN)
- **Libraries**:
  - `TensorFlow`/`Keras` for building and training the CNN model
  - `OpenCV` for real-time facial expression recognition
  - `Streamlit` for creating the user interface
  - `Pandas`, `NumPy` for data handling and preprocessing
- **Programming Language**: Python