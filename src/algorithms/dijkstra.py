import networkx as nx
import heapq

def dijkstra(graphe, source):
    """
    Implémente l'algorithme de Dijkstra pour trouver le chemin le plus court
    depuis la source vers toutes les autres intersections.
    """
    # Dictionnaire pour stocker les distances minimales
    distances = {noeud: float('inf') for noeud in graphe.nodes}
    distances[source] = 0

    # Liste de priorité pour explorer les noeuds
    pq = [(0, source)]  # (distance, noeud)

    while pq:
        dist, noeud_courant = heapq.heappop(pq)

        # Ignorer les noeuds qui ont été déjà explorés avec une distance plus courte
        if dist > distances[noeud_courant]:
            continue

        # Explorer les voisins
        for voisin in graphe.neighbors(noeud_courant):
            poids = graphe[noeud_courant][voisin]['poids']
            nouvelle_distance = dist + poids

            # Si une distance plus courte est trouvée, mettre à jour
            if nouvelle_distance < distances[voisin]:
                distances[voisin] = nouvelle_distance
                heapq.heappush(pq, (nouvelle_distance, voisin))

    return distances
