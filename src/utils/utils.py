# -*- coding: utf-8 -*-
"""
Fonctions utilitaires générales pour le projet de recherche du plus court chemin.
"""

import collections
import itertools
import numpy as np
import os
import time
import matplotlib.pyplot as plt


def paires_consecutives(iterable):
    """Génère des paires consécutives à partir d'un itérable: s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def difference_ensembles(seq0, seq1):
    """Retourne la différence entre 2 séquences sous forme de liste."""
    return list(set(seq0) - set(seq1))


def vers_one_hot(indices, valeur_max, axe=-1):
    """Convertit des indices en vecteurs one-hot."""
    one_hot = np.eye(valeur_max)[indices]
    if axe not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axe)
    return one_hot


def obtenir_dict_noeuds(graphe, attr):
    """Retourne un dictionnaire de paires nœud:attribut à partir d'un graphe."""
    return {k: v[attr] for k, v in graphe.nodes.items()}


def creer_repertoire(chemin):
    """Crée un répertoire s'il n'existe pas déjà."""
    if not os.path.exists(chemin):
        os.makedirs(chemin)
    return chemin


def softmax_prob_derniere_dim(x):
    """Calcule la probabilité softmax sur la dernière dimension."""
    e = np.exp(x)
    return e[:, -1] / np.sum(e, axis=-1)


def enregistrer_metriques(iterations, pertes_tr, corrects_tr, resolus_tr, 
                         pertes_ge, corrects_ge, resolus_ge, chemin_fichier):
    """Enregistre les métriques d'entraînement et de généralisation dans un fichier."""
    with open(chemin_fichier, 'w') as f:
        f.write("iteration,perte_entrainement,correct_entrainement,resolus_entrainement,"
                "perte_generalisation,correct_generalisation,resolus_generalisation\n")
        for i, (it, ptr, ctr, rtr, pge, cge, rge) in enumerate(zip(
                iterations, pertes_tr, corrects_tr, resolus_tr, 
                pertes_ge, corrects_ge, resolus_ge)):
            f.write(f"{it},{ptr},{ctr},{rtr},{pge},{cge},{rge}\n")


def tracer_resultats(iterations, pertes_tr, corrects_tr, resolus_tr, 
                    pertes_ge, corrects_ge, resolus_ge, chemin_fichier=None):
    """Trace les courbes de résultats d'entraînement et de généralisation."""
    fig = plt.figure(figsize=(18, 5))
    x = np.array(iterations)
    
    # Perte
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(x, pertes_tr, "k", label="Entraînement")
    ax.plot(x, pertes_ge, "k--", label="Généralisation")
    ax.set_title("Perte au cours de l'entraînement")
    ax.set_xlabel("Itération d'entraînement")
    ax.set_ylabel("Perte (entropie croisée binaire)")
    ax.legend()
    
    # Correct
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(x, corrects_tr, "k", label="Entraînement")
    ax.plot(x, corrects_ge, "k--", label="Généralisation")
    ax.set_title("Fraction correcte au cours de l'entraînement")
    ax.set_xlabel("Itération d'entraînement")
    ax.set_ylabel("Fraction de nœuds/arêtes corrects")
    ax.legend()
    
    # Résolu
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(x, resolus_tr, "k", label="Entraînement")
    ax.plot(x, resolus_ge, "k--", label="Généralisation")
    ax.set_title("Fraction résolue au cours de l'entraînement")
    ax.set_xlabel("Itération d'entraînement")
    ax.set_ylabel("Fraction d'exemples résolus")
    ax.legend()
    
    plt.tight_layout()
    
    if chemin_fichier:
        plt.savefig(chemin_fichier)
    
    return fig 