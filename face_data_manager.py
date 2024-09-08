import cv2
import os
import pandas as pd
import uuid
import face_recognition

class FaceDataManager:
    def __init__(self):
        # Directory for saving images
        self.save_dir = 'saved_data'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Data file for saving ID, gender, and age
        self.data_file = 'person_data.csv'
        if os.path.isfile(self.data_file) and os.stat(self.data_file).st_size > 0:
            self.df = pd.read_csv(self.data_file)
        else:
            self.df = pd.DataFrame(columns=['ID', 'Gender', 'Age', 'ImagePath'])

        # Dictionary to store face encodings and IDs
        self.face_encodings = {}
        self.face_ids = {}

    def save_image(self, image, face_id):
        img_filename = f'{self.save_dir}/person_{face_id}.jpg'
        cv2.imwrite(img_filename, image)
        return img_filename

    def add_data(self, face_id, gender, age, img_filename):
        new_row = pd.DataFrame({'ID': [face_id], 'Gender': [gender], 'Age': [age], 'ImagePath': [img_filename]})
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.data_file, index=False)

    def find_or_add_person(self, face_encoding, gender, age):
        # Compare with existing face encodings
        for person_id, encoding in self.face_encodings.items():
            if face_recognition.compare_faces([encoding], face_encoding)[0]:
                return person_id
        
        # If no match, create a new ID
        face_id = str(uuid.uuid4())
        self.face_encodings[face_id] = face_encoding
        return face_id
