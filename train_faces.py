import face_recognition
import os
import pickle

def encode_faces(directory):
    face_encodings = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                face_encodings[filename] = encodings[0]

    with open("encodings.pickle", "wb") as f:
        pickle.dump(face_encodings, f)

if __name__ == "__main__":
    dataset_directory = "dataset"  # where you store your face images
    encode_faces(dataset_directory)