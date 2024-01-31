#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os


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

# mettre trente sous dossier nommé avec le nom de la classe
# mettre dedans les photos concerné 
create_imagesFolder()

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