from ..graph_utils import creer_reseau_routier, afficher_reseau_routier
from ..algorithms.dijkstra import dijkstra
from ..algorithms.bellman_ford import bellman_ford
import sys
sys.path.append('/c:/Users/LEGION 5/Documents/1A MIS/Projet integré/Python projet/shortest-path-project')

def tester_algorithmes():
    """
    Teste les algorithmes de Dijkstra et Bellman-Ford sur le réseau routier.
    Affiche les distances calculées par chaque algorithme.
    """
    # Créer le réseau routier
    G = creer_reseau_routier()
    afficher_reseau_routier(G)
    
    source = 1  # Choisir l'Intersection A comme point de départ

    # Tester Dijkstra
    distances_dijkstra = dijkstra(G, source)
    print("\nDistances avec Dijkstra:")
    for noeud, distance in distances_dijkstra.items():
        print(f"Intersection {noeud}: {distance} km")

    # Tester Bellman-Ford
    distances_bellman = bellman_ford(G, source)
    if distances_bellman is not None:
        print("\nDistances avec Bellman-Ford:")
        for noeud, distance in distances_bellman.items():
            print(f"Intersection {noeud}: {distance} km")


# Exécution du test
if __name__ == "__main__":
    tester_algorithmes()
