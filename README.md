# Audience Emotion Analyzer

The Audience Emotion Analyzer analyzes a video and tries to detect faces in a crowd, classify the emotion of each face and then creates a report in `csv` format.

The app might be a little slow to start the first time, but the model is cached using `st.cache_resource` so the following runs should be faster as long as you don't restart the app.

## Files
`requirements.txt` is a file to install the necessary Python libraries with `pip -r`.

`haarcascade_frontalface_default.xml` is the model for OpenCV to detect faces.

`modelv1.keras` is a model trained to classify emotions. (It isn't very good at it, unfortunately.)

`app.py` is a `streamlit` app where a user can upload a video and have it analyzed.

`src/analyzer.py` is the source file for the `Analyzer` class.
