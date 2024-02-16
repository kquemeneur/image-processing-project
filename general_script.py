import pathlib
from pathlib import Path

import cv2
import pandas
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2 as cv
import pickle
import random
from matplotlib import pyplot as plt

# Path to the folder containing your images

IMAGES_FOLDER_PATH = './media/img_align_celeba'
FACES_PICKLE_PATH = './faces.pkl'
CELEB_IDENTITY_FILE = "./celeb_mappings/identity_CelebA.txt"
CELEB_ATTR_LIST_FILE = "./celeb_mappings/list_attr_celeba.txt"
CELEB_LANDMARKS_FILE = "./celeb_mappings/list_landmarks_align_celeba.txt"

"""
Méthode pour récupérer les eigen faces moyenne pour chaque classe (visage de célébrité)
"""
def get_mean_eigen_faces_per_class(identity_faces: list):
    meanEigenList = []
    eigenVectorsList = []
    print(identity_faces.Identity)
    output_df = pandas.DataFrame(columns=["Identity", "MeanFace", "EigenVectors"])
    for index, vec_array in enumerate(identity_faces["Vector"]):
        mean, eigenVectors = cv2.PCACompute(np.array(vec_array, dtype=np.float32), mean=None, maxComponents=20)
        identity = identity_faces["Identity"][index]
        output_df.at[index, "Identity"] = identity
        output_df.at[index, "MeanFace"] = mean
        output_df.at[index, "EigenVectors"] = eigenVectors
        """plt.imshow(mean.reshape(100, 100, 3).astype(int))
        plt.show()
        plt.close()
        fig, axes = plt.subplots(nrows=4, ncols=5,
                                 figsize=(10, 8))
        normalized_vectors = (eigenVectorsList * 255)
        for i, ax in enumerate(axes.flat):
            ax.imshow(normalized_vectors[i].reshape(100, 100, 3))
            ax.axis('off')  # Turn off axis
            ax.set_title(f'Image {i + 1}')  # Set title for each image
        plt.tight_layout()
        plt.show()
        plt.close()"""
    return output_df

"""
Méthode pour supprimer les fichiers pickle
"""
def clear_pickle_files():
    if Path(FACES_PICKLE_PATH).exists():
        os.remove(FACES_PICKLE_PATH)
"""
Classe FaceDataset correspondant à un ensemble de données relatives aux images)
"""
class FaceDataset:
    # celeb_number : number of most represented celeb to choose for the dataset
    def __init__(self, celeb_number: int):
        self.faces = pd.DataFrame(columns=["Filename", "Identity", "Attributes", "Vector"])
        self.n = celeb_number
        self.initialize()
    """
    Fonction d'initialisation du dataset
    """
    def initialize(
            self,
            images_folder: Path = IMAGES_FOLDER_PATH,
            celeb_identity_file: Path = CELEB_IDENTITY_FILE,
            celeb_attr_file: Path = CELEB_ATTR_LIST_FILE,
            celeb_landmarks_file: Path = CELEB_LANDMARKS_FILE,
            faces_pickle: Path = FACES_PICKLE_PATH):
        # If a pickle exist for the images, retrieve from the pickle
        if Path(faces_pickle).is_file():
            self.retrieve_faces_from_pickle(faces_pickle)
        else: # else load images and dump them in a pickle
            self.retrieve_faces_from_scratch(
                images_folder,
                celeb_identity_file,
                celeb_attr_file,
                celeb_landmarks_file)
    """
    Récupérer les données du dataset via un fichier pickle
    """
    def retrieve_faces_from_pickle(
            self,
            faces_pkl_path: Path = FACES_PICKLE_PATH):
        # open and load pickle files
        with open(faces_pkl_path, 'rb') as f:
            loaded_faces = pickle.load(f)

        self.faces = loaded_faces

    """
    Récupérer les données du dataset à partir du dossier d'images et des fichiers d'attributs
    """
    def retrieve_faces_from_scratch(
            self,
            images_folder: Path = IMAGES_FOLDER_PATH,
            celeb_identity_file: Path = CELEB_IDENTITY_FILE,
            celeb_attr_file: Path = CELEB_ATTR_LIST_FILE,
            celeb_landmarks_file: Path = CELEB_LANDMARKS_FILE):
        # retrieve celeb identity file, attr file and landmarks file
        identity_file = Path(celeb_identity_file)
        # attr_file = Path(celeb_attr_file)
        # landmarks_file = Path(celeb_landmarks_file)

        most_represented_filenames_list, most_represented_identity_list = self.extract_most_represented_faces(identity_file)
        self.faces["Filename"] = most_represented_filenames_list
        self.faces["Identity"] = most_represented_identity_list
        # retrieve and associate attributes to the previously extracted images
        # attr_matrix_cols = pd.read_csv(attr_file, delim_whitespace=True, nrows=1).columns
        # attr_matrix = pd.read_csv(attr_file, delim_whitespace=True, usecols=attr_matrix_cols)
        # faces_attr = []
        # for file in self.faces["Filename"]:
        #    faces_attr.append(attr_matrix.loc[attr_matrix['Filename'] == file].values[0][1:])
        # self.faces['Attributes'] = faces_attr

        # load images vectors from filename and associate it to the right face
        for index, filename in enumerate(self.faces["Filename"]):
            # re-construct file path
            img_path = os.path.join(images_folder, filename)
            try:
                # retrieve image vector
                img = cv.imread(img_path)
                img = cv.resize(img, (100, 100))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                height, width, bgr = img.shape
                # reshape to 1D vector
                img = img.reshape(height * width * bgr)
                # store into self.faces dataframe
                self.faces.at[index, "Vector"] = img
            except Exception as e:
                print(f"Error loading image: {img_path} - {e}")

        # save faces in pickle file
        with open(FACES_PICKLE_PATH, 'wb') as file:
            pickle.dump(self.faces, file)

    """
    Associate image with their identity and extract most represented ones
    """
    def extract_most_represented_faces(self, identity_file):
        # process the identity file to have a matrix which we will use to classify the faces
        identity_matrix = pd.read_csv(identity_file, sep=" ", header=None)
        identity_matrix.columns = ["FileName", "Label"]

        # find and extract as a list which labels are more represented
        most_represented_labels = identity_matrix['Label'].value_counts()[:self.n].index.values
        most_represented_extract = identity_matrix[identity_matrix['Label'].isin(most_represented_labels)]
        # most_represented_pictures : array of most represented faces filenames we want to extract
        most_represented_filenames_list = most_represented_extract["FileName"].tolist()
        most_represented_identity_list = most_represented_extract["Label"].tolist()

        return most_represented_filenames_list, most_represented_identity_list

    """
    Séparer le jeu de données en un ensemble d'apprentissage et un ensemble de test
    Retourne les deux ensembles
    """
    def split(self):
        # TODO : check if dataset is initialized, if not return exception
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.faces.Vector,
            self.faces.Identity,
            test_size=0.3,
            random_state=42)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def sort_vectors_by_identity(self):
        vectors_by_identity = self.faces.groupby('Identity')["Vector"].apply(list).reset_index()
        return vectors_by_identity

""" ------------------ ~ MAIN ALGORITHM ~ ------------------ """
# clear_pickle_files()
face_dataset = FaceDataset(30)

vectors_by_identity = face_dataset.sort_vectors_by_identity()

# Calculate mean and eigenFaces for all face vectors
face_vectors_as_numpy_arr = np.array(face_dataset.faces["Vector"].tolist(), dtype=np.uint8)

mean, eigenVectors = cv2.PCACompute(face_vectors_as_numpy_arr, mean=None, maxComponents=5)

# Calculate average eigen face for each class (celebrity)
# eigen_faces_per_class = get_mean_eigen_faces_per_class(vectors_by_identity)
# with open(Path('./mean_eigen_faces.pkl'), 'wb') as file:
#    pickle.dump(eigen_faces_per_class, file)

with open(Path('./mean_eigen_faces.pkl'), 'rb') as f:
    eigen_faces_per_class = pickle.load(f)

# Pick random face image
rand_index = random.randint(0, len(face_dataset.faces.Identity))
random_face = face_dataset.faces.iloc[rand_index]

print(len(random_face.Vector))
# projection = np.dot(np.array(random_face.Vector, dtype=np.uint8), mean)
# plt.imshow(np.array(random_face.Vector, dtype=np.uint8).reshape(100, 100, 3))
# plt.show()
# print(projection.shape)

# Compare random image to averages eigen faces via distance euclidienne
def predict_nearest_identity(eigen_faces, input_face):
    min_distance = None
    nearest_label = None
    for index, identity in enumerate(eigen_faces["Identity"]):
        distance = np.linalg.norm(eigen_faces.at[index, "MeanFace"].astype(int) - input_face)
        if index == 0:
            min_distance = distance
        elif distance < min_distance:
            min_distance = distance
            nearest_label = identity
    # Retrieve the closest images and compare
    return nearest_label, min_distance

# TODO: Train/test split and get prediction score
X_train, X_test, Y_train, Y_test = face_dataset.split()
print(X_train)
predictions = []
for index, face in enumerate(X_train):
    predicted_label, min_dist = predict_nearest_identity(eigen_faces_per_class, face)
    predictions.append(predicted_label)
print(predictions[:10])
print(Y_train[:10])

