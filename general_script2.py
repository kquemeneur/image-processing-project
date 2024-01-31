#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import pandas as pd
import pathlib
from pathlib import Path

CELEB_IDENTITY_FILE = "./celeb_mappings/identity_CelebA.txt"
identity_file = Path(CELEB_IDENTITY_FILE)

#recuperer la liste du top 30 
def extract_most_represented_faces( identity_file):
        # process the identity file to have a matrix which we will use to classify the faces
        identity_matrix = pd.read_csv(identity_file, sep=" ", header=None)
        identity_matrix.columns = ["FileName", "Label"]

        # find and extract as a list which labels are more represented
        most_represented_labels = identity_matrix['Label'].value_counts()[:30].index.values
        most_represented_extract = identity_matrix[identity_matrix['Label'].isin(most_represented_labels)]
        # most_represented_pictures : array of most represented faces filenames we want to extract
        most_represented_filenames_list = most_represented_extract["FileName"].tolist()
        #most_represented_identity_list = most_represented_extract["Label"].tolist()

        return most_represented_filenames_list
# faire un dossier top 30
def create_imagesFolder():
    #verifier si le dossier top 30 n'existe pas déja
    if not os.path.exists("top30"):
        os.mkdir("top30")
    
        #creer les sous dossiers
        for i in range ( 1, 30 +1):
            nom_sous_dossier =  f"top30/classe_{i}"
            try:
                os.mkdir(nom_sous_dossier)
            except FileExistsError:
                print(f"Le sous-dossier '{nom_sous_dossier}' existe déjà.")
    else:
        print(f"Le dossier existe déjà.")
            
create_imagesFolder()
most_represented_filenames = extract_most_represented_faces(identity_file)
print(f"liste '{most_represented_filenames}'")

"""
#data pre-processing and data augmentation (transform image to create new one)
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


#Generate news pictures
img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
"""