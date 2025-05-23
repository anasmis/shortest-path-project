# -*- coding: utf-8 -*-
"""
Point d'entrée principal pour le projet de recherche du plus court chemin.
"""


import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from graph_nets import utils_np
from graph_nets import utils_tf
from src.utils.constants import (GRAINE_ALEATOIRE, TAILLE_LOT_ENTRAINEMENT, 
                               TAILLE_LOT_GENERALISATION, NOMBRE_NOEUDS_MIN_MAX_ENTRAINEMENT,
                               NOMBRE_NOEUDS_MIN_MAX_GENERALISATION, NOMBRE_ETAPES_TRAITEMENT_ENTRAINEMENT,
                               NOMBRE_ETAPES_TRAITEMENT_GENERALISATION, NOMBRE_ITERATIONS_ENTRAINEMENT,
                               TAUX_APPRENTISSAGE, THETA_DEFAUT, CHEMIN_MODELE, CHEMIN_RESULTATS)
from src.utils.utils import creer_repertoire, tracer_resultats, enregistrer_metriques
from src.graph_utils import generer_graphes_networkx
from src.data_utils import (creer_placeholders, creer_feed_dict, calculer_precision,
                          rendre_tout_executable_dans_session)
from src.gnn import ModelePlusCourtChemin
from src.visualisation import visualiser_exemples_graphes, visualiser_predictions


def analyser_arguments():
    """Analyse les arguments de ligne de commande.
    
    Returns:
        args: Les arguments analysés.
    """
    parser = argparse.ArgumentParser(description='Projet de recherche du plus court chemin avec GNN')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'visualize'],
                        help='Mode d\'exécution: train, evaluate ou visualize')
    parser.add_argument('--iterations', type=int, default=NOMBRE_ITERATIONS_ENTRAINEMENT,
                        help='Nombre d\'itérations d\'entraînement')
    parser.add_argument('--taille_lot', type=int, default=TAILLE_LOT_ENTRAINEMENT,
                        help='Taille du lot pour l\'entraînement')
    parser.add_argument('--theta', type=float, default=THETA_DEFAUT,
                        help='Paramètre theta pour la génération des graphes')
    parser.add_argument('--chemin_modele', type=str, default=CHEMIN_MODELE,
                        help='Chemin pour sauvegarder/charger le modèle')
    parser.add_argument('--chemin_resultats', type=str, default=CHEMIN_RESULTATS,
                        help='Chemin pour sauvegarder les résultats')
    parser.add_argument('--graine', type=int, default=GRAINE_ALEATOIRE,
                        help='Graine aléatoire pour la reproductibilité')
    return parser.parse_args()


def entrainer_modele(args):
    """Entraîne le modèle.
    
    Args:
        args: Les arguments de ligne de commande.
    """
    # Créer les répertoires nécessaires
    chemin_modele = args.chemin_modele
    chemin_resultats = args.chemin_resultats
    creer_repertoire(os.path.dirname(chemin_modele))
    creer_repertoire(chemin_resultats)
    
    # Initialiser la graine aléatoire
    rand = np.random.RandomState(seed=args.graine)
    tf.random.set_seed(args.graine)
    
    # Paramètres du modèle
    nombre_etapes_traitement_tr = NOMBRE_ETAPES_TRAITEMENT_ENTRAINEMENT
    nombre_etapes_traitement_ge = NOMBRE_ETAPES_TRAITEMENT_GENERALISATION
    
    # Paramètres d'entraînement
    nombre_iterations = args.iterations
    theta = args.theta
    taille_lot_tr = args.taille_lot
    taille_lot_ge = TAILLE_LOT_GENERALISATION
    
    # Créer les placeholders
    input_ph, target_ph = creer_placeholders(rand, taille_lot_tr,
                                            NOMBRE_NOEUDS_MIN_MAX_ENTRAINEMENT, theta)
    
    # Créer le modèle
    modele = ModelePlusCourtChemin()
    
    # Configurer l'entraînement
    output_ops_tr = modele(input_ph, nombre_etapes_traitement_tr)
    loss_op_tr, step_op = modele.configurer_entrainement(
        target_ph, output_ops_tr, nombre_etapes_traitement_tr, TAUX_APPRENTISSAGE)
    
    # Configurer l'évaluation
    output_ops_ge = modele(input_ph, nombre_etapes_traitement_ge)
    loss_op_ge = modele.configurer_evaluation(target_ph, output_ops_ge)
    
    # Variables pour suivre les métriques
    derniere_iteration = 0
    iterations_enregistrees = []
    pertes_tr = []
    corrects_tr = []
    resolus_tr = []
    pertes_ge = []
    corrects_ge = []
    resolus_ge = []
    
    # Boucle d'entraînement
    for iteration in range(nombre_iterations):
        # Générer les données d'entraînement
        feed_dict, graphes_bruts = creer_feed_dict(
            rand, taille_lot_tr, NOMBRE_NOEUDS_MIN_MAX_ENTRAINEMENT, theta, input_ph, target_ph)
        
        # Étape d'entraînement
        valeurs_entrainement = step_op(feed_dict)
        
        # Évaluer périodiquement
        if iteration % 100 == 0:
            # Évaluer sur les données d'entraînement
            valeurs_test = {
                "target": target_ph,
                "loss": loss_op_tr,
                "outputs": output_ops_tr
            }
            valeurs_test = {k: v(feed_dict) for k, v in valeurs_test.items()}
            
            correct_tr, resolu_tr = calculer_precision(
                valeurs_test["target"], valeurs_test["outputs"][-1], utiliser_aretes=True)
            
            # Évaluer sur les données de généralisation
            feed_dict_ge, _ = creer_feed_dict(
                rand, taille_lot_ge, NOMBRE_NOEUDS_MIN_MAX_GENERALISATION, theta, input_ph, target_ph)
            
            valeurs_ge = {
                "target": target_ph,
                "loss": loss_op_ge,
                "outputs": output_ops_ge
            }
            valeurs_ge = {k: v(feed_dict_ge) for k, v in valeurs_ge.items()}
            
            correct_ge, resolu_ge = calculer_precision(
                valeurs_ge["target"], valeurs_ge["outputs"][-1], utiliser_aretes=True)
            
            # Enregistrer les métriques
            iterations_enregistrees.append(iteration)
            pertes_tr.append(valeurs_test["loss"])
            corrects_tr.append(correct_tr)
            resolus_tr.append(resolu_tr)
            pertes_ge.append(valeurs_ge["loss"])
            corrects_ge.append(correct_ge)
            resolus_ge.append(resolu_ge)
            
            # Afficher les résultats
            print(f"Iteration {iteration}:")
            print(f"  Entraînement - Perte: {pertes_tr[-1]:.4f}, Correct: {corrects_tr[-1]:.4f}, Résolu: {resolus_tr[-1]:.4f}")
            print(f"  Généralisation - Perte: {pertes_ge[-1]:.4f}, Correct: {corrects_ge[-1]:.4f}, Résolu: {resolus_ge[-1]:.4f}")
            
            # Sauvegarder le modèle
            modele.save_model(chemin_modele)
            
            # Tracer les résultats
            tracer_resultats(iterations_enregistrees, pertes_tr, corrects_tr, resolus_tr,
                           pertes_ge, corrects_ge, resolus_ge, chemin_resultats)
            
            # Enregistrer les métriques
            enregistrer_metriques(iterations_enregistrees, pertes_tr, corrects_tr, resolus_tr,
                                pertes_ge, corrects_ge, resolus_ge, chemin_resultats)
            
            derniere_iteration = iteration


def evaluer_modele(args):
    """Évalue le modèle.
    
    Args:
        args: Les arguments de ligne de commande.
    """
    # Initialiser la graine aléatoire
    rand = np.random.RandomState(seed=args.graine)
    tf.set_random_seed(args.graine)
    
    # Réinitialiser le graphe TensorFlow
    tf.reset_default_graph()
    
    # Paramètres du modèle
    nombre_etapes_traitement_ge = NOMBRE_ETAPES_TRAITEMENT_GENERALISATION
    
    # Paramètres d'évaluation
    theta = args.theta
    taille_lot_ge = TAILLE_LOT_GENERALISATION
    
    # Créer les placeholders
    input_ph, target_ph = creer_placeholders(rand, taille_lot_ge,
                                            NOMBRE_NOEUDS_MIN_MAX_GENERALISATION, theta)
    
    # Instancier le modèle
    modele = ModelePlusCourtChemin()
    
    # Connecter les données au modèle
    output_ops_ge = modele(input_ph, nombre_etapes_traitement_ge)
    
    # Configurer l'évaluation
    _, loss_op_ge = modele.configurer_evaluation(target_ph, output_ops_ge)
    
    # Rendre les graphes exécutables dans la session
    input_ph, target_ph = rendre_tout_executable_dans_session(input_ph, target_ph)
    
    # Créer une session TensorFlow
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Charger le modèle
    modele.charger(sess, args.chemin_modele)
    print(f"Modèle chargé depuis {args.chemin_modele}")
    
    # Évaluer le modèle
    feed_dict, graphes_bruts = creer_feed_dict(
        rand, taille_lot_ge, NOMBRE_NOEUDS_MIN_MAX_GENERALISATION, theta, input_ph, target_ph)
    valeurs_test = sess.run({
        "target": target_ph,
        "loss": loss_op_ge,
        "outputs": output_ops_ge
    },
                           feed_dict=feed_dict)
    correct_ge, resolu_ge = calculer_precision(
        valeurs_test["target"], valeurs_test["outputs"][-1], utiliser_aretes=True)
    
    print("Résultats de l'évaluation:")
    print(f"Perte: {valeurs_test['loss']:.4f}")
    print(f"Précision (nœuds/arêtes): {correct_ge:.4f}")
    print(f"Graphes résolus: {resolu_ge:.4f}")
    
    # Visualiser les prédictions
    cibles = utils_np.graphs_tuple_to_data_dicts(valeurs_test["target"])
    sorties = list(zip(*(utils_np.graphs_tuple_to_data_dicts(valeurs_test["outputs"][i])
                        for i in [-1])))  # Utiliser seulement la dernière étape
    indices_etape = [nombre_etapes_traitement_ge - 1]
    
    fig = visualiser_predictions(graphes_bruts, cibles, sorties, indices_etape)
    plt.savefig(os.path.join(args.chemin_resultats, "predictions.png"))
    plt.show()
    
    # Fermer la session
    sess.close()


def visualiser_graphes(args):
    """Visualise des exemples de graphes.
    
    Args:
        args: Les arguments de ligne de commande.
    """
    # Initialiser la graine aléatoire
    rand = np.random.RandomState(seed=args.graine)
    
    # Générer des graphes
    _, _, graphes = generer_graphes_networkx(
        rand, 15, NOMBRE_NOEUDS_MIN_MAX_ENTRAINEMENT, args.theta)
    
    # Visualiser les graphes
    fig = visualiser_exemples_graphes(graphes)
    plt.savefig(os.path.join(args.chemin_resultats, "exemples_graphes.png"))
    plt.show()


def main():
    """Fonction principale."""
    args = analyser_arguments()
    
    if args.mode == 'train':
        entrainer_modele(args)
    elif args.mode == 'evaluate':
        evaluer_modele(args)
    elif args.mode == 'visualize':
        visualiser_graphes(args)
    else:
        print(f"Mode inconnu: {args.mode}")


if __name__ == "__main__":
    main()
