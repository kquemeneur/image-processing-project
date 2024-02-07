from collections import defaultdict
import heapq
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array 
from PIL import Image

"""Trier le top 30 des classes les plus représentées"""
def get_most_represented_labels(file_path, top_count=30):
    # Lire le fichier texte d'image
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Sauvegarder les classes et leur fréquence d'apparition
    labels_count = defaultdict(int)
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            labels_count[parts[1]] += 1

    # Get the 50 most frequent classes
    most_represented_labels = heapq.nlargest(top_count, labels_count, key=labels_count.get)

    return most_represented_labels

""" **** TEST AFFICHER TOP 30 ****
# Appel de la fonction "top classes"
top_labels = get_most_represented_labels(file_path)

# Afficher le top 30 
for i, labels in enumerate(top_labels, start=1):
    print(i, labels)
"""

"""Deplacer les images correspondant au top 30 dans les sous dossiers"""
def create_folders(top30_folder, top_labels) :
    # Creer un fichier pour chaque classe
    for label in top_labels:
        label_folder = os.path.join(top30_folder, label)
        os.makedirs(label_folder, exist_ok=True)
        
def organize_images_by_labels(file_path, image_folder, top30_folder, top_count=30):
    # Recuper le top 30 des classes
    top_labels = get_most_represented_labels(file_path)

    create_folders(top30_folder, top_labels)
    # Lire le fichier texte et deplacer les images au bon endroit
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            parts = line.split()
            if len(parts) == 2:
                image_name, label = parts
                source_path = os.path.join(image_folder, image_name)
                destination_folder = os.path.join(top30_folder, label)
                if os.path.exists(destination_folder):
                    shutil.copy(source_path, destination_folder)

""" Defition des liens"""
file_path = './media/identity_CelebA.txt'
image_folder = './media/img_align_celeba/'
top30_folder = './media/top30/'

organize_images_by_labels(file_path, image_folder, top30_folder)  

"""Pré-traitement des données"""
#Modifier chaque photo pour créer plus de données
datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def refocus_images(img, zoom):
    img_w, img_h = img.size
    new_w= int(img_w / zoom)
    new_h = int(img_h / zoom)

    left =(img_w - new_w) / 2
    right =(img_w + new_w) / 2
    top =(img_h - new_h) / 2
    bottom =(img_h + new_h) / 2

    #couper l'image pour la recentrer
    refocus_image = img.crop((left,top,right,bottom))
    return refocus_image

def generate_transformed_images(img_path, output_folder, datagen, number_generated=5, zoom=1.2):
    #ouvrir le fichier de l'image
    img = Image.open(img_path)

    #recentrer l'image sur le visage
    refocused_img = refocus_images(img, zoom)

    #convertir les images en une liste 
    x = img_to_array(refocused_img) # taille (3,150,150)
    x = x.reshape((1,) + x.shape) # taille (1,3,150,150)

    #Generer les images transformer
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir=output_folder, save_prefix='generated', save_format='jpg'):
        i += 1
        if i > number_generated:
            break  # eviter la boucle infini

def organize_generated_images_by_labels(input_folder, output_folder):
    #creer le dossier de sortie s'il existe pas encore
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    #
    for subdir, dirs, files in os.walk(input_folder):
        subfolder_o = os.path.join(output_folder,os.path.relpath(subdir, top30_folder))
        if not os.path.exists(subfolder_o):
            os.makedirs(subfolder_o)
        
        for file in files:
            img_path_i = os.path.join(subdir,file)
            
            #generer les images transformer
            generate_transformed_images(img_path_i, subfolder_o, datagen, 2)

input_folder = './media/top30/'
output_folder = './media/top30_generated/'
organize_generated_images_by_labels(input_folder, output_folder)

