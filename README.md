# Projet : Neural Discrete Representation Learning

## Introduction

Dans ce projet, nous avons étudié l'article *Neural Discrete Representation Learning* de Aaron van den Oord et al. (arXiv:1711.00937v2).

Nous avons repris le projet GitHub suivant pour implémenter notre travail :  
[https://github.com/dataflowr/Project-VQ-VAE-Images](https://github.com/dataflowr/Project-VQ-VAE-Images).

## Environnement de travail

Nous avons utilisé **Google Colab** pour exécuter le code, avec un **GPU A100** afin de garantir des temps d'exécution raisonnables pour les expérimentations.

## Processus

1. **Suivi du Repository d'Origine**  
   Nous avons suivi le guide fourni dans le repo pour exécuter le code sur le jeu de données **Cifar10**.

2. **Adaptation au Dataset FashionMNIST**  
   Nous avons adapté le code pour traiter le jeu de données **FashionMNIST**, en modifiant et en utilisant le script `src/FashionMNIST_dataset.py`.

3. **Impact de la Taille de l'Espace Discret K**  
   Nous avons évalué l'impact de la taille de l'espace discret \( K \) en exécutant le code pendant 10 000 itérations, avec \( K \) variant de 39 à 89 par pas de 5. Les résultats ont été stockés dans le fichier `results`.

4. **Décodage du Prior Uniforme**  
   Dans le notebook **Fichier d'exécution.ipynb**, nous avons écrit un code permettant de décoder un prior uniforme après avoir entraîné le modèle sur les datasets **Cifar10** et **FashionMNIST**. Les images obtenues représentent les "textures fondamentales" des deux datasets.

## Perspectives

Pour aller plus loin, nous avons trouvé un repo GitHub implémentant l'entraînement du prior **PixelCNN** :  
[https://github.com/yashgarg98/VQ-VAE/blob/main/vq_vae.ipynb](https://github.com/yashgarg98/VQ-VAE/blob/main/vq_vae.ipynb).  
Cependant, cette implémentation n’a pas été intégrée dans notre projet.
