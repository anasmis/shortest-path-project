# -*- coding: utf-8 -*-
"""
Utilitaires pour la manipulation des graphes dans le projet de recherche du plus court chemin.
"""

import collections
import itertools
import numpy as np
import networkx as nx
from scipy import spatial

from src.utils.constants import NOM_POIDS_DISTANCE
from src.utils.utils import paires_consecutives, difference_ensembles, vers_one_hot, obtenir_dict_noeuds


def generer_graphe(rand,
                  plage_nombre_noeuds,
                  dimensions=2,
                  theta=1000.0,
                  taux=1.0):
    """Crée un graphe connecté.

    Les graphes sont des graphes à seuil géographique, mais avec des arêtes supplémentaires
    via un algorithme d'arbre couvrant minimal, pour garantir que tous les nœuds sont connectés.

    Args:
        rand: Une graine aléatoire pour le générateur de graphes.
        plage_nombre_noeuds: Une séquence [min, max) pour le nombre de nœuds par graphe.
        dimensions: (optionnel) Un nombre entier de dimensions pour les positions. Par défaut = 2.
        theta: (optionnel) Un paramètre de seuil flottant pour le seuil du graphe à seuil
            géographique. Des valeurs élevées (1000+) créent principalement des arbres. Essayez
            20-60 pour de bons non-arbres. Par défaut = 1000.0.
        taux: (optionnel) Un paramètre de taux pour la distribution d'échantillonnage
            exponentielle des poids des nœuds. Par défaut = 1.0.

    Returns:
        Le graphe.
    """
    # Échantillonnage du nombre de nœuds
    nombre_noeuds = rand.randint(*plage_nombre_noeuds)

    # Création du graphe à seuil géographique
    tableau_pos = rand.uniform(size=(nombre_noeuds, dimensions))
    pos = dict(enumerate(tableau_pos))
    poids = dict(enumerate(rand.exponential(taux, size=nombre_noeuds)))
    graphe_geo = nx.geographical_threshold_graph(
        nombre_noeuds, theta, pos=pos, weight=poids)

    # Création d'un arbre couvrant minimal sur les nœuds de graphe_geo
    distances = spatial.distance.squareform(spatial.distance.pdist(tableau_pos))
    i_, j_ = np.meshgrid(range(nombre_noeuds), range(nombre_noeuds), indexing="ij")
    aretes_ponderees = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    graphe_acm = nx.Graph()
    graphe_acm.add_weighted_edges_from(aretes_ponderees, weight=NOM_POIDS_DISTANCE)
    graphe_acm = nx.minimum_spanning_tree(graphe_acm, weight=NOM_POIDS_DISTANCE)
    # Mettre les attributs de nœud de graphe_geo dans graphe_acm
    for i in graphe_acm.nodes():
        graphe_acm.nodes[i].update(graphe_geo.nodes[i])

    # Composer les graphes
    graphe_combine = nx.compose_all((graphe_acm, graphe_geo.copy()))
    # Mettre tous les poids de distance dans les attributs d'arête
    for i, j in graphe_combine.edges():
        graphe_combine.get_edge_data(i, j).setdefault(NOM_POIDS_DISTANCE,
                                                distances[i, j])
    return graphe_combine, graphe_acm, graphe_geo


def ajouter_plus_court_chemin(rand, graphe, longueur_min=1):
    """Échantillonne un plus court chemin de A à B et ajoute des attributs pour l'indiquer.

    Args:
        rand: Une graine aléatoire pour le générateur de graphes.
        graphe: Un graphe `nx.Graph`.
        longueur_min: (optionnel) Un nombre entier minimum d'arêtes dans le plus court
            chemin. Par défaut = 1.

    Returns:
        Le `nx.DiGraph` avec le plus court chemin ajouté.

    Raises:
        ValueError: Tous les plus courts chemins sont en dessous de la longueur minimale
    """
    # Dictionnaire des paires de nœuds vers la longueur de leur plus court chemin
    dict_paire_vers_longueur = {}
    try:
        # Pour la compatibilité avec les versions plus anciennes de networkx
        longueurs = nx.all_pairs_shortest_path_length(graphe).items()
    except AttributeError:
        # Pour la compatibilité avec les versions plus récentes de networkx
        longueurs = list(nx.all_pairs_shortest_path_length(graphe))
    for x, yy in longueurs:
        for y, l in yy.items():
            if l >= longueur_min:
                dict_paire_vers_longueur[x, y] = l
    if max(dict_paire_vers_longueur.values()) < longueur_min:
        raise ValueError("Tous les plus courts chemins sont en dessous de la longueur minimale")
    # Les paires de nœuds qui dépassent la longueur minimale
    paires_noeuds = list(dict_paire_vers_longueur)

    # Calcule les probabilités par paire, pour imposer un échantillonnage uniforme de chaque
    # longueur de plus court chemin
    # Le comptage des paires par longueur
    comptes = collections.Counter(dict_paire_vers_longueur.values())
    prob_par_longueur = 1.0 / len(comptes)
    probabilites = [
        prob_par_longueur / comptes[dict_paire_vers_longueur[x]] for x in paires_noeuds
    ]

    # Choisir les points de départ et d'arrivée
    i = rand.choice(len(paires_noeuds), p=probabilites)
    depart, arrivee = paires_noeuds[i]
    chemin = nx.shortest_path(
        graphe, source=depart, target=arrivee, weight=NOM_POIDS_DISTANCE)

    # Crée un graphe orienté pour stocker le chemin orienté de départ à arrivée
    digraphe = graphe.to_directed()

    # Ajouter les attributs "depart", "arrivee" et "solution" aux nœuds et arêtes
    digraphe.add_node(depart, start=True)
    digraphe.add_node(arrivee, end=True)
    digraphe.add_nodes_from(difference_ensembles(digraphe.nodes(), [depart]), start=False)
    digraphe.add_nodes_from(difference_ensembles(digraphe.nodes(), [arrivee]), end=False)
    digraphe.add_nodes_from(difference_ensembles(digraphe.nodes(), chemin), solution=False)
    digraphe.add_nodes_from(chemin, solution=True)
    aretes_chemin = list(paires_consecutives(chemin))
    digraphe.add_edges_from(difference_ensembles(digraphe.edges(), aretes_chemin), solution=False)
    digraphe.add_edges_from(aretes_chemin, solution=True)

    return digraphe


def graphe_vers_entree_cible(graphe):
    """Renvoie 2 graphes avec des vecteurs de caractéristiques d'entrée et cible pour l'entraînement.

    Args:
        graphe: Une instance `nx.DiGraph`.

    Returns:
        L'instance `nx.DiGraph` d'entrée.
        L'instance `nx.DiGraph` cible.

    Raises:
        ValueError: type de nœud inconnu
    """

    def creer_caracteristique(attr, champs):
        return np.hstack([np.array(attr[champ], dtype=float) for champ in champs])

    champs_noeud_entree = ("pos", "weight", "start", "end")
    champs_arete_entree = ("distance",)
    champs_noeud_cible = ("solution",)
    champs_arete_cible = ("solution",)

    graphe_entree = graphe.copy()
    graphe_cible = graphe.copy()

    longueur_solution = 0
    for indice_noeud, caracteristique_noeud in graphe.nodes(data=True):
        graphe_entree.add_node(
            indice_noeud, features=creer_caracteristique(caracteristique_noeud, champs_noeud_entree))
        noeud_cible = vers_one_hot(
            creer_caracteristique(caracteristique_noeud, champs_noeud_cible).astype(int), 2)[0]
        graphe_cible.add_node(indice_noeud, features=noeud_cible)
        longueur_solution += int(caracteristique_noeud["solution"])
    longueur_solution /= graphe.number_of_nodes()

    for recepteur, emetteur, caracteristiques in graphe.edges(data=True):
        graphe_entree.add_edge(
            emetteur, recepteur, features=creer_caracteristique(caracteristiques, champs_arete_entree))
        arete_cible = vers_one_hot(
            creer_caracteristique(caracteristiques, champs_arete_cible).astype(int), 2)[0]
        graphe_cible.add_edge(emetteur, recepteur, features=arete_cible)

    graphe_entree.graph["features"] = np.array([0.0])
    graphe_cible.graph["features"] = np.array([longueur_solution], dtype=float)

    return graphe_entree, graphe_cible


def generer_graphes_networkx(rand, nombre_exemples, plage_nombre_noeuds, theta):
    """Génère des graphes pour l'entraînement.

    Args:
        rand: Une graine aléatoire (instance np.RandomState).
        nombre_exemples: Nombre total de graphes à générer.
        plage_nombre_noeuds: Un tuple à 2 éléments avec le nombre [min, max) de nœuds par
            graphe. Le nombre de nœuds pour un graphe est échantillonné uniformément dans cette
            plage.
        theta: (optionnel) Un paramètre de seuil flottant pour le seuil du graphe à seuil
            géographique. Par défaut = le nombre de nœuds.

    Returns:
        graphes_entree: La liste des graphes d'entrée.
        graphes_cible: La liste des graphes de sortie.
        graphes: La liste des graphes générés.
    """
    graphes_entree = []
    graphes_cible = []
    graphes = []
    for _ in range(nombre_exemples):
        graphe = generer_graphe(rand, plage_nombre_noeuds, theta=theta)[0]
        graphe = ajouter_plus_court_chemin(rand, graphe)
        graphe_entree, graphe_cible = graphe_vers_entree_cible(graphe)
        graphes_entree.append(graphe_entree)
        graphes_cible.append(graphe_cible)
        graphes.append(graphe)
    return graphes_entree, graphes_cible, graphes
