import pathlib
from pathlib import Path
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2 as cv
import pickle

# Path to the folder containing your images

IMAGES_FOLDER_PATH = './media/img_align_celeba'
FACE_IMAGES_PICKLE_PATH = './face_images.pkl'
FACE_LABELS_PICKLE_PATH = './face_labels.pkl'
CELEB_IDENTITY_FILE = "./celeb_mappings/identity_CelebA.txt"
CELEB_ATTR_LIST_FILE = "./celeb_mappings/list_attr_celeba.txt"
CELEB_LANDMARKS_FILE = "./celeb_mappings/list_landmarks_align_celeba.txt"

def load_images_from_folder_and_file_list(images_folder_path: Path, file_list: list):
    images = []
    labels = []
    for index, filename in enumerate(os.listdir(images_folder_path)):
        if filename in file_list:
            img_path = os.path.join(images_folder_path, filename)
            try:
                img = cv.imread(img_path)
                img = cv.resize(img, (100, 100))  # Resize image as needed
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
                images.append(img)
                labels.append(index)  # Assign label index
            except Exception as e:
                print(f"Error loading image: {img_path} - {e}")
    return np.array(images), np.array(labels)


class FaceDataset:
    # celeb_number : number of most represented celeb to choose for the dataset
    def __init__(self, celeb_number: int):
        self.vectors = []
        self.labels = []
        self.attr = []
        self.landmarks = []
        self.n = celeb_number
        self.initialize()

    # If a pickle exist for the images, retrieve from the pickle, else load images and dump them in a pickle
    def initialize(
            self,
            images_folder: Path = IMAGES_FOLDER_PATH,
            celeb_identity_file: Path = CELEB_IDENTITY_FILE,
            celeb_attr_file: Path = CELEB_ATTR_LIST_FILE,
            celeb_landmarks_file: Path = CELEB_LANDMARKS_FILE,
            face_images_pickle: Path = FACE_IMAGES_PICKLE_PATH,
            face_labels_pickle: Path = FACE_LABELS_PICKLE_PATH):
        if Path(FACE_IMAGES_PICKLE_PATH).is_file() and Path(FACE_LABELS_PICKLE_PATH).is_file():
            self.retrieve_faces_from_pickle(face_images_pickle, face_labels_pickle)
        else:
            self.retrieve_faces_from_scratch(
                images_folder,
                celeb_identity_file,
                celeb_attr_file,
                celeb_landmarks_file)

    def retrieve_faces_from_pickle(
            self,
            images_pkl_path: Path = FACE_IMAGES_PICKLE_PATH,
            labels_pkl_path: Path = FACE_LABELS_PICKLE_PATH):

        with open(images_pkl_path, 'rb') as f:
            face_images_loaded = pickle.load(f)

        with open(labels_pkl_path, 'rb') as f:
            face_labels_loaded = pickle.load(f)

        self.vectors = face_images_loaded
        self.labels = face_labels_loaded

    def retrieve_faces_from_scratch(
            self,
            images_folder: Path = IMAGES_FOLDER_PATH,
            celeb_identity_file: Path = CELEB_IDENTITY_FILE,
            celeb_attr_file: Path = CELEB_ATTR_LIST_FILE,
            celeb_landmarks_file: Path = CELEB_LANDMARKS_FILE):
        # retrieve celeb identity file, attr file and landmarks file
        identity_file = Path(celeb_identity_file)
        attr_file = Path(celeb_attr_file)
        landmarks_file = Path(celeb_landmarks_file)

        # process the identity file to have a matrix which we will use to classify the faces
        identity_matrix = pd.read_csv(identity_file, sep=" ", header=None)
        identity_matrix.columns = ["FileName", "Label"]

        # find and extract as a list which labels are more represented
        most_represented_labels = identity_matrix['Label'].value_counts()[:self.n].index.values
        most_represented_extract = identity_matrix[identity_matrix['Label'].isin(most_represented_labels)]
        most_represented_pictures = most_represented_extract["FileName"].tolist()

        # Load images and labels
        face_images, face_labels = load_images_from_folder_and_file_list(images_folder, most_represented_pictures)
        print(face_images[0].shape)

        images = np.array(face_images)
        print(images.shape)
        n_samples, height, width, bgr = images.shape
        face_images = images.reshape(n_samples, height * width, bgr)

        with open(FACE_IMAGES_PICKLE_PATH, 'wb') as file:
            pickle.dump(face_images, file)
        with open(FACE_LABELS_PICKLE_PATH, 'wb') as file:
            pickle.dump(face_labels, file)

        self.vectors = face_images
        self.labels = face_labels

    def split(self):
        # TODO : check if dataset is initialized, if not return exception
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.vectors,
            self.labels,
            test_size=0.3,
            random_state=42)
        return self.X_train, self.X_test, self.Y_train, self.Y_test


""" ------------------ ~ MAIN ALGORITHM ~ ------------------ """
# TODO extract the most representated celeb pictures to use it as train dataset

face_dataset = FaceDataset(10)
X_train, X_test, Y_train, Y_test = face_dataset.split()
print(len(X_train))

# TODO store images with pickle

# TODO Calculate vector distance (Eigenface)
    # TODO Chose attributes to compare (just the eyes for example)
    # TODO Make average of each class

