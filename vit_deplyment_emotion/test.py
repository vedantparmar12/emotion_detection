import cv2
import torch
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# Define the model name
model_name = "Vedant101/vit-base-patch16-224-finetuned"

# Load the pre-trained emotion classification model
model = ViTForImageClassification.from_pretrained(model_name)
model.load_state_dict(torch.load("vit_emotion_model.pth"))
model.eval()  # Set the model to evaluation mode

# Load the feature extractor
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

# Define the list of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess the face ROI
        inputs = feature_extractor(images=face_roi, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        label = emotion_labels[predicted_class]

        # Display the predicted emotion label on the frame
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()