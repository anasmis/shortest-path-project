# -*- coding: utf-8 -*-
"""
Utilitaires de visualisation pour le projet de recherche du plus court chemin.
"""

import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.utils.utils import softmax_prob_derniere_dim, obtenir_dict_noeuds
from src.utils.constants import (TAILLE_NOEUD, LARGEUR_ARETE, LARGEUR_ARETE_SOLUTION, 
                               COULEUR_NOEUD_DEFAUT, COULEUR_NOEUD_DEPART, COULEUR_NOEUD_ARRIVEE)


class TraceurGraphe(object):
    """Classe pour tracer des graphes avec des chemins les plus courts."""

    def __init__(self, ax, graphe, pos):
        """Initialise le traceur de graphe.
        
        Args:
            ax: L'axe matplotlib sur lequel tracer.
            graphe: Le graphe networkx à tracer.
            pos: Les positions des nœuds.
        """
        self._ax = ax
        self._graphe = graphe
        self._pos = pos
        self._kwargs_dessin_base = dict(G=self._graphe, pos=self._pos, ax=self._ax)
        self._longueur_solution = None
        self._noeuds = None
        self._aretes = None
        self._noeuds_depart = None
        self._noeuds_arrivee = None
        self._noeuds_solution = None
        self._noeuds_solution_intermediaires = None
        self._aretes_solution = None
        self._noeuds_non_solution = None
        self._aretes_non_solution = None
        self._ax.set_axis_off()

    @property
    def longueur_solution(self):
        """Renvoie la longueur de la solution."""
        if self._longueur_solution is None:
            self._longueur_solution = len(self._aretes_solution)
        return self._longueur_solution

    @property
    def noeuds(self):
        """Renvoie les nœuds du graphe."""
        if self._noeuds is None:
            self._noeuds = self._graphe.nodes()
        return self._noeuds

    @property
    def aretes(self):
        """Renvoie les arêtes du graphe."""
        if self._aretes is None:
            self._aretes = self._graphe.edges()
        return self._aretes

    @property
    def noeuds_depart(self):
        """Renvoie les nœuds de départ."""
        if self._noeuds_depart is None:
            self._noeuds_depart = [
                n for n in self.noeuds if self._graphe.nodes[n].get("start", False)
            ]
        return self._noeuds_depart

    @property
    def noeuds_arrivee(self):
        """Renvoie les nœuds d'arrivée."""
        if self._noeuds_arrivee is None:
            self._noeuds_arrivee = [
                n for n in self.noeuds if self._graphe.nodes[n].get("end", False)
            ]
        return self._noeuds_arrivee

    @property
    def noeuds_solution(self):
        """Renvoie les nœuds de la solution."""
        if self._noeuds_solution is None:
            self._noeuds_solution = [
                n for n in self.noeuds if self._graphe.nodes[n].get("solution", False)
            ]
        return self._noeuds_solution

    @property
    def noeuds_solution_intermediaires(self):
        """Renvoie les nœuds intermédiaires de la solution."""
        if self._noeuds_solution_intermediaires is None:
            self._noeuds_solution_intermediaires = [
                n for n in self.noeuds
                if self._graphe.nodes[n].get("solution", False) and
                not self._graphe.nodes[n].get("start", False) and
                not self._graphe.nodes[n].get("end", False)
            ]
        return self._noeuds_solution_intermediaires

    @property
    def aretes_solution(self):
        """Renvoie les arêtes de la solution."""
        if self._aretes_solution is None:
            self._aretes_solution = [
                e for e in self.aretes
                if self._graphe.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._aretes_solution

    @property
    def noeuds_non_solution(self):
        """Renvoie les nœuds qui ne font pas partie de la solution."""
        if self._noeuds_non_solution is None:
            self._noeuds_non_solution = [
                n for n in self.noeuds
                if not self._graphe.nodes[n].get("solution", False)
            ]
        return self._noeuds_non_solution

    @property
    def aretes_non_solution(self):
        """Renvoie les arêtes qui ne font pas partie de la solution."""
        if self._aretes_non_solution is None:
            self._aretes_non_solution = [
                e for e in self.aretes
                if not self._graphe.get_edge_data(e[0], e[1]).get("solution", False)
            ]
        return self._aretes_non_solution

    def _creer_kwargs_dessin(self, **kwargs):
        """Crée des arguments pour les fonctions de dessin."""
        kwargs.update(self._kwargs_dessin_base)
        return kwargs

    def _dessiner(self, fonction_dessin, zorder=None, **kwargs):
        """Dessine en utilisant la fonction de dessin spécifiée."""
        kwargs_dessin = self._creer_kwargs_dessin(**kwargs)
        collection = fonction_dessin(**kwargs_dessin)
        if collection is not None and zorder is not None:
            try:
                # Pour la compatibilité avec l'ancien matplotlib
                collection.set_zorder(zorder)
            except AttributeError:
                # Pour la compatibilité avec le nouveau matplotlib
                collection[0].set_zorder(zorder)
        return collection

    def dessiner_noeuds(self, **kwargs):
        """Dessine les nœuds. Kwargs utiles: nodelist, node_size, node_color, linewidths."""
        if ("node_color" in kwargs and
                isinstance(kwargs["node_color"], collections.Sequence) and
                len(kwargs["node_color"]) in {3, 4} and
                not isinstance(kwargs["node_color"][0],
                               (collections.Sequence, np.ndarray))):
            num_noeuds = len(kwargs.get("nodelist", self.noeuds))
            kwargs["node_color"] = np.tile(
                np.array(kwargs["node_color"])[None], [num_noeuds, 1])
        return self._dessiner(nx.draw_networkx_nodes, **kwargs)

    def dessiner_aretes(self, **kwargs):
        """Dessine les arêtes. Kwargs utiles: edgelist, width."""
        return self._dessiner(nx.draw_networkx_edges, **kwargs)

    def dessiner_graphe(self,
                       taille_noeud=TAILLE_NOEUD,
                       couleur_noeud=COULEUR_NOEUD_DEFAUT,
                       largeur_contour_noeud=1.0,
                       largeur_arete=LARGEUR_ARETE):
        """Dessine le graphe sans mettre en évidence la solution."""
        # Dessiner les nœuds
        self.dessiner_noeuds(
            nodelist=self.noeuds,
            node_size=taille_noeud,
            node_color=couleur_noeud,
            linewidths=largeur_contour_noeud,
            zorder=20)
        # Dessiner les arêtes
        self.dessiner_aretes(edgelist=self.aretes, width=largeur_arete, zorder=10)

    def dessiner_graphe_avec_solution(self,
                                     taille_noeud=TAILLE_NOEUD,
                                     couleur_noeud=COULEUR_NOEUD_DEFAUT,
                                     largeur_contour_noeud=1.0,
                                     largeur_arete=LARGEUR_ARETE,
                                     couleur_depart=COULEUR_NOEUD_DEPART,
                                     couleur_arrivee=COULEUR_NOEUD_ARRIVEE,
                                     largeur_contour_noeud_solution=3.0,
                                     largeur_arete_solution=LARGEUR_ARETE_SOLUTION):
        """Dessine le graphe en mettant en évidence la solution."""
        couleur_bordure_noeud = (0.0, 0.0, 0.0, 1.0)
        collections_noeuds = {}
        # Dessiner les nœuds de départ
        collections_noeuds["noeuds_depart"] = self.dessiner_noeuds(
            nodelist=self.noeuds_depart,
            node_size=taille_noeud,
            node_color=couleur_depart,
            linewidths=largeur_contour_noeud_solution,
            edgecolors=couleur_bordure_noeud,
            zorder=100)
        # Dessiner les nœuds d'arrivée
        collections_noeuds["noeuds_arrivee"] = self.dessiner_noeuds(
            nodelist=self.noeuds_arrivee,
            node_size=taille_noeud,
            node_color=couleur_arrivee,
            linewidths=largeur_contour_noeud_solution,
            edgecolors=couleur_bordure_noeud,
            zorder=90)
        # Dessiner les nœuds intermédiaires de la solution
        if isinstance(couleur_noeud, dict):
            c = [couleur_noeud[n] for n in self.noeuds_solution_intermediaires]
        else:
            c = couleur_noeud
        collections_noeuds["noeuds_solution_intermediaires"] = self.dessiner_noeuds(
            nodelist=self.noeuds_solution_intermediaires,
            node_size=taille_noeud,
            node_color=c,
            linewidths=largeur_contour_noeud_solution,
            edgecolors=couleur_bordure_noeud,
            zorder=80)
        # Dessiner les arêtes de la solution
        collections_noeuds["aretes_solution"] = self.dessiner_aretes(
            edgelist=self.aretes_solution, width=largeur_arete_solution, zorder=70)
        # Dessiner les nœuds qui ne font pas partie de la solution
        if isinstance(couleur_noeud, dict):
            c = [couleur_noeud[n] for n in self.noeuds_non_solution]
        else:
            c = couleur_noeud
        collections_noeuds["noeuds_non_solution"] = self.dessiner_noeuds(
            nodelist=self.noeuds_non_solution,
            node_size=taille_noeud,
            node_color=c,
            linewidths=largeur_contour_noeud,
            edgecolors=couleur_bordure_noeud,
            zorder=20)
        # Dessiner les arêtes qui ne font pas partie de la solution
        collections_noeuds["aretes_non_solution"] = self.dessiner_aretes(
            edgelist=self.aretes_non_solution, width=largeur_arete, zorder=10)
        # Définir le titre comme la longueur de la solution
        self._ax.set_title("Longueur de la solution: {}".format(self.longueur_solution))
        return collections_noeuds


def visualiser_exemples_graphes(graphes, nombre_max=16):
    """Visualise des exemples de graphes.
    
    Args:
        graphes: Liste de graphes à visualiser.
        nombre_max: Nombre maximum de graphes à visualiser.
        
    Returns:
        fig: La figure matplotlib.
    """
    num = min(len(graphes), nombre_max)
    w = 3
    h = int(np.ceil(num / w))
    fig = plt.figure(figsize=(w * 4, h * 4))
    fig.clf()
    for j, graphe in enumerate(graphes[:num]):
        ax = fig.add_subplot(h, w, j + 1)
        pos = obtenir_dict_noeuds(graphe, "pos")
        traceur = TraceurGraphe(ax, graphe, pos)
        traceur.dessiner_graphe_avec_solution()
    plt.tight_layout()
    return fig


def visualiser_predictions(graphes_bruts, cibles, sorties, indices_etape, min_c=0.3, max_graphes=6):
    """Visualise les prédictions du modèle.
    
    Args:
        graphes_bruts: Liste des graphes bruts.
        cibles: Liste des graphes cibles.
        sorties: Liste des sorties du modèle.
        indices_etape: Liste des indices d'étape à visualiser.
        min_c: Valeur minimale pour la couleur.
        max_graphes: Nombre maximum de graphes à visualiser.
        
    Returns:
        fig: La figure matplotlib.
    """
    num_graphes = len(graphes_bruts)
    num_etapes = len(indices_etape)
    h = min(num_graphes, max_graphes)
    w = num_etapes + 1
    fig = plt.figure(figsize=(18, h * 3))
    fig.clf()
    
    for j, (graphe, cible, sortie) in enumerate(zip(graphes_bruts, cibles, sorties)):
        if j >= h:
            break
        pos = obtenir_dict_noeuds(graphe, "pos")
        verite_terrain = cible["nodes"][:, -1]
        
        # Vérité terrain
        iax = j * (1 + num_etapes) + 1
        ax = fig.add_subplot(h, w, iax)
        traceur = TraceurGraphe(ax, graphe, pos)
        couleur = {}
        for i, n in enumerate(traceur.noeuds):
            couleur[n] = np.array([1.0 - verite_terrain[i], 0.0, verite_terrain[i], 1.0]) * (1.0 - min_c) + min_c
        traceur.dessiner_graphe_avec_solution(couleur_noeud=couleur)
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            ax.set_facecolor([0.9] * 3 + [1.0])
        except AttributeError:
            ax.set_axis_bgcolor([0.9] * 3 + [1.0])
        ax.grid(None)
        ax.set_title("Vérité terrain\nLongueur de la solution: {}".format(
            traceur.longueur_solution))
        
        # Prédiction
        for k, outp in enumerate(sortie):
            iax = j * (1 + num_etapes) + 2 + k
            ax = fig.add_subplot(h, w, iax)
            traceur = TraceurGraphe(ax, graphe, pos)
            couleur = {}
            prob = softmax_prob_derniere_dim(outp["nodes"])
            for i, n in enumerate(traceur.noeuds):
                couleur[n] = np.array([1.0 - prob[n], 0.0, prob[n], 1.0]) * (1.0 - min_c) + min_c
            traceur.dessiner_graphe_avec_solution(couleur_noeud=couleur)
            ax.set_title("Prédiction du modèle\nÉtape {:02d} / {:02d}".format(
                indices_etape[k] + 1, indices_etape[-1] + 1))
    
    plt.tight_layout()
    return fig 