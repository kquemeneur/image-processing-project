from collections import defaultdict
import heapq
import os
import shutil
""" Defition des liens"""
file_path = './media/identity_CelebA.txt'
image_folder = './media/img_align_celeba/'
top30_folder = './media/top30_folder/'

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
    top_labels = heapq.nlargest(top_count, labels_count, key=labels_count.get)

    return top_labels

# Appel de la fonction "top classes"
top_labels = get_most_represented_labels(file_path)

# Afficher le top 30 
for i, labels in enumerate(top_labels, start=1):
    print(i, labels)

"""Deplacer les images correspondant au top 30 dans les sous dossiers"""
def create_folders(top30_folder, top_labels) :
    # Creer un fichier pour chaque classe
    for label in top_labels:
        label_folder = os.path.join(top30_folder, label)
        os.makedirs(label_folder, exist_ok=True)
        
def organize_images_by_labels(file_path, image_folder, top30_folder, top_count=30):
    # Recuper le top 30 des classes
    top_label = get_most_represented_labels(file_path, top_count=30)

    create_folders(top30_folder, top_label)
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


organize_images_by_labels(file_path, image_folder, top30_folder, top_count=30)  