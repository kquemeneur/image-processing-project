# Projet de reconnaissance faciale dans le cadre du module *Analyse et traitement d'images*
par Apolline Chénais, Léane Diraison et Katell Quéméneur
## Pré-requis
Une version fonctionnelle de python3 et d'anaconda sur votre machine.
## Instructions
Suivez dans l'ordre les instructions suivantes afin de pouvoir exécuter les scripts qui ont été mis en place.
### Récupération du code source
- Cloner le repo git : `git clone <SSH-OR-HTTP-LINK>`
- Se déplacer dans le dossier *image-processing-project* : `cd image-processing-project`
### Initialisation de l'environnement python anaconda  
- Créer l'environement : `conda create --name image_processing_env python=3`
- Installer les dépendances nécessaires : `conda install --name image_processing_env numpy scipy scikit-learn scikit-image opencv matplotlib keras tensorflow pytorch pandas`
- Activer l'environnement : `conda activate image_processing_env`
### Copier les images du jeu de données CelebA
- Copier dans le dossier media/ l'ensemble des images du jeu de données CelebA aligné.
Les images doivent être dans **un sous dossier qui doit impérativement se nommer** *img_align_celeba* on peut alors retrouver les images en suivant le chemin : *./media/img_align_celeba* à partir de la racine du projet.
### Executer les scripts
- Pour exécuter le script sur la partie 1 du projet : `python3 script_partie1.py`
- Pour exécuter le script sur la partie 2 du projet : `python3 script_partie2.py`