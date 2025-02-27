import networkx as nx
import random as rnd
def creer_reseau_routier():
    """
    Crée un réseau routier de base avec des intersections comme noeuds
    et des routes comme arêtes. Chaque arête a un poids (distance ou temps).
    """
    # Initialiser un graphe dirigé
    G = nx.DiGraph()

    # Ajouter des noeuds (intersections)
    G.add_node(1, nom="Intersection A")
    G.add_node(2, nom="Intersection B")
    G.add_node(3, nom="Intersection C")
    G.add_node(4, nom="Intersection D")

    # Ajouter des arêtes (routes) avec des poids (distances ou temps)
    G.add_edge(1, 2, poids=rnd.randint(0,25))  # Route de A à B, 10 km
    G.add_edge(2, 3, poids=rnd.randint(0,25))  # Route de B à C, 15 km
    G.add_edge(3, 4, poids=rnd.randint(0,25))  # Route de C à D, 10 km
    G.add_edge(4, 1, poids=rnd.randint(0,25))  # Route de D à A, 20 km

    # Retourner le graphe
    return G


def afficher_reseau_routier(G):
    """
    Affiche le réseau routier (noeuds et arêtes) avec leurs poids.
    """
    print("Réseau routier:")
    for noeud, donnees in G.nodes(data=True):
        print(f"Noeud {noeud}: {donnees['nom']}")

    print("\nArêtes avec poids:")
    for u, v, donnees in G.edges(data=True):
        print(f"Route de l'Intersection {u} à l'Intersection {v}, Distance: {donnees['poids']} km")

# Exemple d'utilisation
if __name__ == "__main__":
    G = creer_reseau_routier()
    afficher_reseau_routier(G)
