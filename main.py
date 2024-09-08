import cv2
import numpy as np
from keras.models import model_from_json
import pandas as pd
import os
import uuid

# Load the Emotion Detector model
with open("emotiondetectorf.json", "r") as json_file:
    emotion_model = model_from_json(json_file.read())
emotion_model.load_weights("emotiondetectorf.h5")

# Load the Age and Gender Detector model
with open("agedetectorf.json", "r") as json_file:
    age_gender_model = model_from_json(json_file.read())
age_gender_model.load_weights("agedetectorf.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Feature extraction function for emotion model
def extract_features(image, model_input_shape):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (model_input_shape[1], model_input_shape[0]))
    feature = np.array(image)
    feature = feature.reshape(1, *model_input_shape)
    return feature / 255.0

# Directory for saving images
save_dir = 'saved_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Data file for saving ID, gender, and age
data_file = 'person_data.csv'
if os.path.isfile(data_file) and os.stat(data_file).st_size > 0:
    df = pd.read_csv(data_file)
else:
    df = pd.DataFrame(columns=['ID', 'Gender', 'Age', 'ImagePath'])

# Initialize ID tracker and dictionary for faces
person_id = 1
face_dict = {}

webcam = cv2.VideoCapture(0)

emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
gender_labels = {0: 'Male', 1: 'Female'}

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No faces detected")
    
    for (x, y, w, h) in faces:
        face_id = str(uuid.uuid4())  # Generate a unique ID for each face
        face_image = gray[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_image = cv2.resize(face_image, (48, 48))

        # Emotion prediction
        img = extract_features(face_image, (48, 48, 1))
        emotion_pred = emotion_model.predict(img)
        emotion_label = emotion_labels[emotion_pred.argmax()]

        # Age and gender prediction
        img_age_gender = extract_features(face_image, (48, 48, 1))
        age_gender_pred = age_gender_model.predict(img_age_gender)

        predicted_gender_index = int(age_gender_pred[0].item())
        predicted_age = int(age_gender_pred[1].item())

        predicted_gender = gender_labels[predicted_gender_index]

        # Save the image with a unique ID
        img_filename = f'{save_dir}/person_{face_id}.jpg'
        cv2.imwrite(img_filename, face_image)

        # Append new data
        new_row = pd.DataFrame({'ID': [face_id], 'Gender': [predicted_gender], 'Age': [predicted_age], 'ImagePath': [img_filename]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(data_file, index=False)

        # Display emotion, gender, and age
        cv2.putText(frame, f'{emotion_label}, {predicted_gender}, Age: {predicted_age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Output", frame)
    
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


