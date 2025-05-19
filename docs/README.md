# Documentation du Projet de Recherche du Plus Court Chemin

Ce document fournit une documentation détaillée sur le projet de recherche du plus court chemin utilisant des réseaux de neurones de graphes (GNN).

## Table des matières

1. [Introduction](#introduction)
2. [Fondements théoriques](#fondements-théoriques)
3. [Architecture du modèle](#architecture-du-modèle)
4. [Structure du projet](#structure-du-projet)
5. [Utilisation](#utilisation)
6. [Résultats](#résultats)
7. [Références](#références)

## Introduction

Le problème du plus court chemin est un problème fondamental en théorie des graphes. Il consiste à trouver le chemin le plus court entre deux nœuds d'un graphe. Les algorithmes traditionnels comme Dijkstra ou A* résolvent ce problème de manière exacte, mais ils doivent être exécutés à chaque fois qu'un nouveau problème de plus court chemin se présente.

Ce projet explore une approche alternative utilisant des réseaux de neurones de graphes (GNN) pour apprendre à résoudre ce problème. L'avantage principal est qu'une fois entraîné, le modèle peut prédire les plus courts chemins sans avoir à exécuter un algorithme complet, ce qui peut être plus rapide pour certaines applications.

## Fondements théoriques

### Problème du plus court chemin

Étant donné un graphe G = (V, E) où V est l'ensemble des nœuds et E l'ensemble des arêtes, et deux nœuds s, t ∈ V, le problème du plus court chemin consiste à trouver un chemin P de s à t tel que la somme des poids des arêtes dans P soit minimale.

### Réseaux de neurones de graphes (GNN)

Les GNN sont des modèles de deep learning conçus pour opérer sur des données structurées en graphes. Contrairement aux réseaux de neurones traditionnels qui traitent des données tabulaires, séquentielles ou en grille, les GNN peuvent traiter des données avec une topologie arbitraire.

Le principe fondamental des GNN est le passage de messages entre les nœuds du graphe. Chaque nœud agrège les informations de ses voisins, met à jour sa représentation, puis transmet cette information mise à jour à ses voisins au prochain pas de temps.

## Architecture du modèle

Notre modèle utilise l'architecture "Encode-Process-Decode" :

1. **Encodeur** : Encode indépendamment les attributs des arêtes, des nœuds et globaux.
2. **Cœur** : Effectue N étapes de traitement (passage de messages). L'entrée du Cœur est la concaténation de la sortie de l'Encodeur et de la sortie précédente du Cœur.
3. **Décodeur** : Décode indépendamment les attributs des arêtes, des nœuds et globaux à chaque étape de passage de messages.

```
                    Hidden(t)   Hidden(t+1)
                       |            ^
          *---------*  |  *------*  |  *---------*
          |         |  |  |      |  |  |         |
Input --->| Encodeur|  *->| Cœur |--*->| Décodeur|---> Output(t)
          |         |---->|      |     |         |
          *---------*     *------*     *---------*
```

Le modèle est entraîné par apprentissage supervisé. Les graphes d'entrée sont générés de manière procédurale, et les graphes de sortie ont la même structure avec les nœuds et les arêtes du plus court chemin étiquetés (en utilisant des vecteurs one-hot à 2 éléments).

## Structure du projet

Le projet est organisé comme suit :

```
.
├── data/                  # Données générées pour l'entraînement et l'évaluation
├── docs/                  # Documentation du projet
│   └── README.md          # Ce document
├── src/                   # Code source
│   ├── utils/             # Fonctions utilitaires
│   │   ├── __init__.py    # Initialisation du package utils
│   │   ├── constants.py   # Constantes du projet
│   │   └── utils.py       # Utilitaires généraux
│   ├── data_utils.py      # Utilitaires pour la manipulation des données
│   ├── graph_utils.py     # Utilitaires pour la manipulation des graphes
│   ├── gnn.py             # Implémentation du réseau de neurones de graphes
│   ├── main.py            # Point d'entrée principal
│   └── visualisation.py   # Fonctions de visualisation
├── tests/                 # Tests unitaires et d'intégration
├── README.md              # Présentation générale du projet
└── requirements.txt       # Dépendances du projet
```

## Utilisation

### Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Entraînement du modèle

```bash
python src/main.py --mode train --iterations 10000 --taille_lot 32 --theta 20
```

Options disponibles :
- `--mode` : Mode d'exécution (train, evaluate, visualize)
- `--iterations` : Nombre d'itérations d'entraînement
- `--taille_lot` : Taille du lot pour l'entraînement
- `--theta` : Paramètre theta pour la génération des graphes
- `--chemin_modele` : Chemin pour sauvegarder/charger le modèle
- `--chemin_resultats` : Chemin pour sauvegarder les résultats
- `--graine` : Graine aléatoire pour la reproductibilité

### Évaluation du modèle

```bash
python src/main.py --mode evaluate
```

### Visualisation des graphes

```bash
python src/main.py --mode visualize
```

## Résultats

Après entraînement, le modèle est capable de prédire les plus courts chemins dans des graphes similaires à ceux sur lesquels il a été entraîné. Les performances sont mesurées par :

1. **Précision des nœuds/arêtes** : Fraction des nœuds et arêtes correctement étiquetés comme faisant partie ou non du plus court chemin.
2. **Graphes résolus** : Fraction des graphes pour lesquels le modèle a correctement identifié l'ensemble du plus court chemin.

Le modèle est également évalué sur sa capacité à généraliser à des graphes plus grands que ceux sur lesquels il a été entraîné.

## Références

1. Battaglia, P. W., et al. (2018). Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261.
2. Gilmer, J., et al. (2017). Neural message passing for quantum chemistry. In International Conference on Machine Learning (pp. 1263-1272).
3. Veličković, P., et al. (2018). Graph attention networks. In International Conference on Learning Representations.
4. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to algorithms. MIT press.
5. DeepMind. (2018). Graph Nets library. https://github.com/deepmind/graph_nets 