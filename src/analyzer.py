import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
import skimage as ski
import av
import streamlit as st

emotion_labels = list(map(str.lower, ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']))

def load_face_detector(cascade_path = './haarcascade_frontalface_default.xml'):
     return cv2.CascadeClassifier(cascade_path)

def initialize_results_df():
     return pd.DataFrame(columns=["frame"] + emotion_labels + ["x", "y", "width", "height"])

def preprocess_face(face_region):
     resized = ski.transform.resize(face_region, (48, 48))
     normalized = resized.astype('float32') / 255.0
     array = keras.utils.img_to_array(normalized)
     return np.expand_dims(array, axis=0)

def predict_emotion(model, face, threshold=0.5):
     prediction = model.predict(face, verbose=0)[0]
     return prediction > threshold

def append_results(df, frame_index, prediction, x, y, w, h):
     if prediction.sum() > 0:
          data = np.concatenate([[frame_index], prediction, [x, y, w, h]])
          row = pd.Series(data, index=df.columns)
          return pd.concat([df, row.to_frame().T])
     return df

def analyze(file, model, skip=1, confidence=0.5, show_faces=False):
    detector = load_face_detector()
    results_df = initialize_results_df()
    screenshots = []

    container = av.open(file, mode='r')
    stream = container.streams.video[0]

    for i, frame in enumerate(container.decode(stream)):        
            rgb = frame.to_rgb().to_ndarray()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]

                if i % skip == 0 and np.sum(roi) !=0:
                    face = tf.convert_to_tensor(preprocess_face(roi))
                    prediction = predict_emotion(model, face, threshold=confidence)

                    results_df = append_results(results_df, i, prediction, x, y, w, h)

                    if show_faces:
                        screenshots.append(roi)

    container.close()
    
    if show_faces:
        st.image(screenshots, caption=[f"Face {i}" for i in range(len(screenshots))], clamp=True)

    return results_df