# Recherche du Plus Court Chemin avec des Réseaux de Neurones de Graphes

Ce projet implémente un modèle de réseau de neurones de graphes (GNN) pour résoudre le problème du plus court chemin dans un graphe. Le modèle est entraîné pour prédire les nœuds et les arêtes qui font partie du plus court chemin entre deux nœuds donnés.

## Description du Projet

Le problème du plus court chemin est un problème fondamental en théorie des graphes qui consiste à trouver le chemin le plus court entre deux nœuds d'un graphe. Ce projet utilise des réseaux de neurones de graphes pour apprendre à résoudre ce problème de manière automatique.

Le modèle est basé sur l'architecture "Encode-Process-Decode" :
- **Encodeur** : Encode indépendamment les attributs des arêtes, des nœuds et globaux
- **Cœur** : Effectue N étapes de traitement (passage de messages)
- **Décodeur** : Décode indépendamment les attributs des arêtes, des nœuds et globaux

## Structure du Projet

```
.
├── data/                  # Données générées pour l'entraînement et l'évaluation
├── docs/                  # Documentation du projet
├── src/                   # Code source
│   ├── utils/             # Fonctions utilitaires
│   ├── algorithms/        # Implémentations d'algorithmes
│   ├── data_utils.py      # Utilitaires pour la manipulation des données
│   ├── graph_utils.py     # Utilitaires pour la manipulation des graphes
│   ├── gnn.py             # Implémentation du réseau de neurones de graphes
│   └── main.py            # Point d'entrée principal
├── tests/                 # Tests unitaires et d'intégration
└── requirements.txt       # Dépendances du projet
```

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

```bash
# Exécuter l'entraînement
python src/main.py --mode train

# Évaluer le modèle
python src/main.py --mode evaluate

# Visualiser les résultats
python src/main.py --mode visualize
```

## Fonctionnalités

- Génération de graphes aléatoires pour l'entraînement
- Calcul du plus court chemin avec l'algorithme de Dijkstra
- Entraînement d'un GNN pour prédire le plus court chemin
- Visualisation des résultats et des prédictions du modèle
- Évaluation de la généralisation sur des graphes plus grands

## Auteurs

Anas AIT ALI
