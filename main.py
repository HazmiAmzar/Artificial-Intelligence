# Import the Libraries
import os
import cv2
import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Title
st.title("Music Recommendation based on Facial Expressions using Deep Learning")
app_mode = st.sidebar.selectbox('Select Page',['Home Page','Real-time camera','Image']) #three pages

#face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Load the model
model = tf.keras.models.load_model("cnnmodel.keras")

#read song dataset
music = pd.read_csv("C:/Users/ACER SWIFT X/Desktop/music recommendation/data_moods.csv")

emotion_dict = {0: "Neutral", 1: "Happy", 2: "Surprise", 3: "Sad", 4: "Angry", 5: "Disgust", 6: "Fear"}

def Recommend_Songs(maxindex):
    if maxindex == 'Disgust':
        # Filter the music DataFrame based on mood
        Play = music[music['mood'] == 'Sad']
    
    elif maxindex in ['Happy', 'Sad']:
        # Filter the music DataFrame based on mood
        Play = music[music['mood'] == 'Happy']
    
    elif maxindex in ['Fear', 'Angry']:
        # Filter the music DataFrame based on mood
        Play = music[music['mood'] == 'Calm']
    
    elif maxindex in ['Surprise', 'Neutral']:
        # Filter the music DataFrame based on mood
        Play = music[music['mood'] == 'Energetic']

    # Sort the filtered DataFrame by popularity
    Play = Play.sort_values(by="popularity", ascending=False)

    # Select columns to display (excluding the 'mood' column)
    Play_to_display = Play[['name', 'artist', 'album']]

    # Select the top 5 rows and reset the index
    Play_to_display = Play_to_display.reset_index(drop=True)

    # Display the selected columns
    st.dataframe(Play_to_display)

if app_mode == 'Home Page':
    st.title('Home page :')  

elif app_mode == 'Real-time camera':    
    st.title('Real-time camera :')
    st.subheader('Your face have to be in the frame to be detected.')
    #st.sidebar.header("Informations about the client :")

    image = st.camera_input(":blue[Capture Image : ] ")

    if image:
        # Convert the image to OpenCV format
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]

            # Resize
            roi_resized = cv2.resize(roi_gray_frame, (48, 48))

            # Expand dimensions to match model input shape
            roi_input = np.expand_dims(roi_resized, axis=0)

            # predict the emotions
            emotion_prediction = model.predict(roi_input)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the frame with the detected faces and emotion labels
        st.image(frame, channels="BGR")

        print()
        st.write("This is the list of musics suitable for emotions: " + emotion_dict[maxindex])
        Recommend_Songs(emotion_dict[maxindex])

elif app_mode == 'Image':
    st.title('Image :')
    upload = st.file_uploader("Upload image:", type= ["png", "jpg", "jpeg"])
    
    #Check if upload is not None before displaying the image
    if upload is not None:
        # Read image file
        image_bytes = upload.getvalue()
        orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) #this image is in BGR form

        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        st.image(rgb_image, caption='Uploaded Image in RGB.', use_column_width=True)
        
        #Convert the frame to grayscale for face detection
        image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        #st.image(image)

        # Detect faces in the frame
        faces = face_detector.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray_frame = image[y:y + h, x:x + w]

            # Resize
            roi_resized = cv2.resize(roi_gray_frame, (48, 48))
            #st.image(roi_resized)
            #roi_resized_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)

            # Expand dimensions to match model input shape
            roi_input = np.expand_dims(roi_resized, axis=0)

            #image = cv2.resize(image, (48, 48))
            # Expand dimensions to match model input shape
            #img_input = np.expand_dims(image, axis=0)

            # Make prediction
            emotion_prediction = model.predict(roi_input)
            maxindex = int(np.argmax(emotion_prediction))
        st.write("Prediction:", emotion_dict[maxindex])
        print()
        st.write("This is the list of musics suitable for emotions: " + emotion_dict[maxindex])
        Recommend_Songs(emotion_dict[maxindex])

    else:
        st.write("Please upload an image.")    
     
