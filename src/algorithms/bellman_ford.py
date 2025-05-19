import networkx as nx
import heapq

def bellman_ford(graphe, source):
    """
    Implémente l'algorithme de Bellman-Ford pour trouver le chemin le plus court
    depuis la source vers toutes les autres intersections, même avec des poids négatifs.
    """
    # Initialiser les distances
    distances = {noeud: float('inf') for noeud in graphe.nodes}
    distances[source] = 0

    # Relaxation des arêtes
    for _ in range(len(graphe.nodes) - 1):
        for u, v, donnees in graphe.edges(data=True):
            poids = donnees['poids']
            if distances[u] + poids < distances[v]:
                distances[v] = distances[u] + poids

    # Vérification des cycles de poids négatifs
    for u, v, donnees in graphe.edges(data=True):
        poids = donnees['poids']
        if distances[u] + poids < distances[v]:
            print("Le graphe contient un cycle de poids négatifs")
            return None

    return distances
