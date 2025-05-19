# -*- coding: utf-8 -*-
"""
Constantes utilisées dans le projet de recherche du plus court chemin.
"""

# Nom de l'attribut de poids pour les distances entre les nœuds
NOM_POIDS_DISTANCE = "distance"

# Paramètres par défaut pour la génération de graphes
THETA_DEFAUT = 20  # Valeurs élevées (1000+) génèrent des arbres, 20-60 pour de bons graphes non-arbres
DIMENSIONS_DEFAUT = 2  # Dimensions pour les positions des nœuds
TAUX_DEFAUT = 1.0  # Paramètre de taux pour la distribution exponentielle des poids des nœuds

# Paramètres pour l'entraînement
GRAINE_ALEATOIRE = 1  # Graine pour la reproductibilité
TAILLE_LOT_ENTRAINEMENT = 32  # Taille du lot pour l'entraînement
TAILLE_LOT_GENERALISATION = 100  # Taille du lot pour l'évaluation de la généralisation
NOMBRE_NOEUDS_MIN_MAX_ENTRAINEMENT = (8, 17)  # Plage du nombre de nœuds pour l'entraînement
NOMBRE_NOEUDS_MIN_MAX_GENERALISATION = (16, 33)  # Plage du nombre de nœuds pour la généralisation
NOMBRE_ETAPES_TRAITEMENT_ENTRAINEMENT = 10  # Nombre d'étapes de traitement pour l'entraînement
NOMBRE_ETAPES_TRAITEMENT_GENERALISATION = 10  # Nombre d'étapes de traitement pour la généralisation
NOMBRE_ITERATIONS_ENTRAINEMENT = 10000  # Nombre d'itérations d'entraînement
TAUX_APPRENTISSAGE = 1e-3  # Taux d'apprentissage pour l'optimiseur

# Paramètres pour la visualisation
TAILLE_NOEUD = 200  # Taille des nœuds dans les visualisations
LARGEUR_ARETE = 1.0  # Largeur des arêtes dans les visualisations
LARGEUR_ARETE_SOLUTION = 3.0  # Largeur des arêtes de la solution dans les visualisations
COULEUR_NOEUD_DEFAUT = (0.4, 0.8, 0.4)  # Couleur par défaut des nœuds (vert)
COULEUR_NOEUD_DEPART = "w"  # Couleur du nœud de départ (blanc)
COULEUR_NOEUD_ARRIVEE = "k"  # Couleur du nœud d'arrivée (noir)

# Chemins des fichiers
CHEMIN_MODELE = "modeles/modele_plus_court_chemin"  # Chemin pour sauvegarder le modèle
CHEMIN_RESULTATS = "resultats/"  # Répertoire pour sauvegarder les résultats 