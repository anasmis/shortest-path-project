# -*- coding: utf-8 -*-
"""
Utilitaires pour la manipulation des données dans le projet de recherche du plus court chemin.
"""

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np

from src.graph_utils import generer_graphes_networkx


def creer_placeholders(rand, taille_lot, plage_nombre_noeuds, theta):
    """Crée des placeholders pour l'entraînement et l'évaluation du modèle.

    Args:
        rand: Une graine aléatoire (instance np.RandomState).
        taille_lot: Nombre total de graphes par lot.
        plage_nombre_noeuds: Un tuple à 2 éléments avec le nombre [min, max) de nœuds par
            graphe. Le nombre de nœuds pour un graphe est échantillonné uniformément dans cette
            plage.
        theta: Un paramètre de seuil flottant pour le seuil du graphe à seuil
            géographique. Par défaut = le nombre de nœuds.

    Returns:
        input_ph: Les placeholders du graphe d'entrée, sous forme de namedtuple de graphe.
        target_ph: Les placeholders du graphe cible, sous forme de namedtuple de graphe.
    """
    # Créer des données d'exemple pour inspecter les tailles de vecteurs
    graphes_entree, graphes_cible, _ = generer_graphes_networkx(
        rand, taille_lot, plage_nombre_noeuds, theta)
    
    # Convertir les graphes NetworkX en tenseurs TensorFlow
    input_ph = utils_tf.placeholders_from_networkxs(graphes_entree)
    target_ph = utils_tf.placeholders_from_networkxs(graphes_cible)
    
    return input_ph, target_ph


def creer_feed_dict(rand, taille_lot, plage_nombre_noeuds, theta, input_ph, target_ph):
    """Crée un dictionnaire d'alimentation pour une session TensorFlow.

    Args:
        rand: Une graine aléatoire (instance np.RandomState).
        taille_lot: Nombre total de graphes par lot.
        plage_nombre_noeuds: Un tuple à 2 éléments avec le nombre [min, max) de nœuds par
            graphe. Le nombre de nœuds pour un graphe est échantillonné uniformément dans cette
            plage.
        theta: Un paramètre de seuil flottant pour le seuil du graphe à seuil
            géographique. Par défaut = le nombre de nœuds.
        input_ph: Les placeholders du graphe d'entrée.
        target_ph: Les placeholders du graphe cible.

    Returns:
        feed_dict: Un dictionnaire d'alimentation pour une session TensorFlow.
        graphes_bruts: Les graphes NetworkX bruts utilisés pour créer le feed_dict.
    """
    # Générer les graphes
    graphes_entree, graphes_cible, graphes_bruts = generer_graphes_networkx(
        rand, taille_lot, plage_nombre_noeuds, theta)
    
    # Convertir les graphes en tenseurs
    input_graphs = utils_np.networkxs_to_graphs_tuple(graphes_entree)
    target_graphs = utils_np.networkxs_to_graphs_tuple(graphes_cible)
    
    # Créer le dictionnaire d'alimentation
    feed_dict = {
        input_ph: input_graphs,
        target_ph: target_graphs
    }
    
    return feed_dict, graphes_bruts


def calculer_metriques(target, output):
    """Calcule les métriques de précision pour les prédictions du modèle.

    Args:
        target: Le graphe cible.
        output: Le graphe de sortie du modèle.

    Returns:
        correct: La fraction de nœuds/arêtes correctement étiquetés.
        resolu: La fraction d'exemples complètement correctement étiquetés.
    """
    # Convertir les tenseurs en tableaux NumPy
    target_np = utils_np.graphs_tuple_to_data_dicts(target)
    output_np = utils_np.graphs_tuple_to_data_dicts(output)
    
    corrects = []
    resolus = []
    
    for t, o in zip(target_np, output_np):
        # Calculer la précision pour les nœuds
        nodes_correct = np.mean(np.argmax(t["nodes"], axis=-1) == np.argmax(o["nodes"], axis=-1))
        
        # Calculer la précision pour les arêtes
        edges_correct = np.mean(np.argmax(t["edges"], axis=-1) == np.argmax(o["edges"], axis=-1))
        
        # Combiner les précisions
        correct = np.mean([nodes_correct, edges_correct])
        resolu = np.all(np.argmax(t["nodes"], axis=-1) == np.argmax(o["nodes"], axis=-1)) and \
                np.all(np.argmax(t["edges"], axis=-1) == np.argmax(o["edges"], axis=-1))
        
        corrects.append(correct)
        resolus.append(resolu)
    
    return np.mean(corrects), np.mean(resolus)


def creer_operations_perte(target_op, output_ops):
    """Crée des opérations de perte pour l'entraînement.

    Args:
        target_op: L'opération cible.
        output_ops: Les opérations de sortie.

    Returns:
        loss_ops: Les opérations de perte.
    """
    loss_ops = [
        tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
        tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
        for output_op in output_ops
    ]
    return loss_ops


def rendre_tout_executable_dans_session(*args):
    """Permet à un itérable de graphes TF d'être produit à partir d'une session en tant que graphes NP."""
    return [utils_tf.make_runnable_in_session(a) for a in args] 