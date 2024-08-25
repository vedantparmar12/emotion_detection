import streamlit as st
import cv2
import torch
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor

@st.cache_resource
def load_model():
    model_name = "Vedant101/vit-base-patch16-224-finetuned"
    model = ViTForImageClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load("vit_emotion_model.pth"))
    model.eval()
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    return model, feature_extractor

def process_frame(frame, face_classifier, model, feature_extractor, emotion_labels):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        face_roi = frame[y:y+h, x:x+w]
        
        inputs = feature_extractor(images=face_roi, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        label = emotion_labels[predicted_class]

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def main():
    st.title("Real-time Emotion Detection")

    model, feature_extractor = load_model()

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    st.write("Click the button below to start the webcam and detect emotions!")

    if st.button("Start Webcam"):
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.write("Failed to capture frame from webcam. Check your webcam connection.")
                break

            frame = process_frame(frame, face_classifier, model, feature_extractor, emotion_labels)
            stframe.image(frame, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()

if __name__ == "__main__":
    main()