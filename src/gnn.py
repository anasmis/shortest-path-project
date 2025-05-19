# -*- coding: utf-8 -*-
"""
Définition du modèle de réseau de neurones de graphes pour le problème du plus court chemin.
"""

import tensorflow as tf
from graph_nets.demos import models


class ModelePlusCourtChemin(tf.keras.Model):
    """Modèle de réseau de neurones de graphes pour le problème du plus court chemin.
    
    Le modèle est basé sur l'architecture "Encode-Process-Decode" :
    - Un réseau "Encodeur" qui encode indépendamment les attributs des arêtes, 
      des nœuds et globaux.
    - Un réseau "Cœur" qui effectue N étapes de traitement (passage de messages).
      L'entrée du Cœur est la concaténation de la sortie de l'Encodeur et de la 
      sortie précédente du Cœur.
    - Un réseau "Décodeur" qui décode indépendamment les attributs des arêtes, 
      des nœuds et globaux, à chaque étape de passage de messages.
    
    Schéma de l'architecture:
    
                        Hidden(t)   Hidden(t+1)
                           |            ^
              *---------*  |  *------*  |  *---------*
              |         |  |  |      |  |  |         |
    Input --->| Encodeur|  *->| Cœur |--*->| Décodeur|---> Output(t)
              |         |---->|      |     |         |
              *---------*     *------*     *---------*
    """
    
    def __init__(self, taille_sortie_arete=2, taille_sortie_noeud=2):
        """Initialise le modèle.
        
        Args:
            taille_sortie_arete: Taille de sortie pour les arêtes (par défaut: 2 pour la classification binaire).
            taille_sortie_noeud: Taille de sortie pour les nœuds (par défaut: 2 pour la classification binaire).
        """
        super(ModelePlusCourtChemin, self).__init__()
        self.taille_sortie_arete = taille_sortie_arete
        self.taille_sortie_noeud = taille_sortie_noeud
        self.modele = models.EncodeProcessDecode(
            edge_output_size=taille_sortie_arete,
            node_output_size=taille_sortie_noeud
        )
    
    def call(self, inputs, num_processing_steps):
        """Appelle le modèle sur les entrées.
        
        Args:
            inputs: Les entrées du modèle (graphe).
            num_processing_steps: Nombre d'étapes de traitement.
            
        Returns:
            Une liste de sorties, une par étape de traitement.
        """
        return self.modele(inputs, num_processing_steps)
    
    def configurer_entrainement(self, target_ph, output_ops_tr, nombre_etapes_traitement_tr,
                              taux_apprentissage=1e-3):
        """Configure l'entraînement du modèle.
        
        Args:
            target_ph: Les placeholders cibles.
            output_ops_tr: Les opérations de sortie pour l'entraînement.
            nombre_etapes_traitement_tr: Nombre d'étapes de traitement pour l'entraînement.
            taux_apprentissage: Taux d'apprentissage pour l'optimiseur.
            
        Returns:
            loss_op_tr: L'opération de perte pour l'entraînement.
            step_op: L'opération d'étape d'optimisation.
        """
        # Perte d'entraînement
        loss_ops_tr = [
            tf.keras.losses.categorical_crossentropy(target_ph.nodes, output_op.nodes) +
            tf.keras.losses.categorical_crossentropy(target_ph.edges, output_op.edges)
            for output_op in output_ops_tr
        ]
        # Perte à travers les étapes de traitement
        loss_op_tr = sum(loss_ops_tr) / nombre_etapes_traitement_tr
        
        # Optimiseur
        optimizer = tf.keras.optimizers.Adam(learning_rate=taux_apprentissage)
        step_op = optimizer.minimize(loss_op_tr, self.trainable_variables)
        
        return loss_op_tr, step_op
    
    def configurer_evaluation(self, target_ph, output_ops_ge):
        """Configure l'évaluation du modèle.
        
        Args:
            target_ph: Les placeholders cibles.
            output_ops_ge: Les opérations de sortie pour l'évaluation.
            
        Returns:
            loss_ops_ge: Les opérations de perte pour l'évaluation.
            loss_op_ge: L'opération de perte finale pour l'évaluation.
        """
        # Perte de test/généralisation
        loss_ops_ge = [
            tf.keras.losses.categorical_crossentropy(target_ph.nodes, output_op.nodes) +
            tf.keras.losses.categorical_crossentropy(target_ph.edges, output_op.edges)
            for output_op in output_ops_ge
        ]
        # Perte de l'étape de traitement finale
        loss_op_ge = loss_ops_ge[-1]
        
        return loss_ops_ge, loss_op_ge
    
    def save_model(self, filepath):
        """Sauvegarde le modèle.
        
        Args:
            filepath: Chemin où sauvegarder le modèle.
        """
        self.save_weights(filepath)
        
    def load_model(self, filepath):
        """Charge le modèle.
        
        Args:
            filepath: Chemin d'où charger le modèle.
        """
        self.load_weights(filepath)
