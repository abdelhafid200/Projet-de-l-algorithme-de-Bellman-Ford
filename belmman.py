from flask import Flask, jsonify, request, render_template
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import matplotlib

# Utilisez le backend non interactif Agg
matplotlib.use('Agg')

from flask import Flask, jsonify, request, render_template
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64

import json
import math
import re
from urllib.request import urlopen
from flask import Flask, jsonify, request, render_template
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import matplotlib
from io import BytesIO
# Utilisez le backend non interactif Agg
matplotlib.use('Agg')

from flask import Flask, jsonify, request, render_template
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
app = Flask(__name__)

# Création d'un graphe de test
G = nx.Graph()

@app.route('/get_neighbors/<int:node_id>', methods=['GET'])
def get_neighbors(node_id):
    neighbors = list(G.neighbors(node_id))
    return jsonify(neighbors)



@app.route('/traverse_graph', methods=['POST'])
def traverse_graph():
    data = request.get_json()
    start_node = data['start_node']

    # Utilisation d'une simple traversée en profondeur (DFS)
    traversal_path = list(nx.dfs_edges(G, source=start_node))

    return jsonify(traversal_path)




@app.route('/')
def plot_graph():
    pos = nx.kamada_kawai_layout(G)# Définir une disposition pour les nœuds du graphe
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=300, node_color='skyblue', font_size=10)

    # Sauvegarder la figure dans une mémoire tampon
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convertir la figure en base64 pour l'afficher dans l'HTML
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return render_template('plot.html', image_data=data)




def draw_graph_from_predecessor_matrix(predecessor_matrix):
    # Créer un graphe dirigé
    G = nx.DiGraph()

    # Ajouter les nœuds au graphe
    num_nodes = len(predecessor_matrix)
    G.add_nodes_from(range(num_nodes))

    # Ajouter les arêtes au graphe en utilisant la matrice des prédécesseurs
    for i in range(num_nodes):
        for j in range(num_nodes):
            if predecessor_matrix[i][j] == -1:
                G.add_edge(j, i)  # Inverser l'orientation des arêtes pour représenter les prédécesseurs

    # Dessiner le graphe
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=700, node_color='skyblue', font_size=10)

    # Sauvegarder la figure dans une mémoire tampon
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Convertir l'image en base64
    image_data = base64.b64encode(buf.read()).decode('utf-8')

    # Récupérer les orientations des arêtes avec les coordonnées des nœuds
    orientations = []
    for edge in G.edges():
        start_node = pos[edge[0]]
        end_node = pos[edge[1]]
        orientations.append(([start_node[0], start_node[1]], [end_node[0], end_node[1]]))

    return {'image_data': image_data, 'orientations': orientations}






@app.route('/draw_graph_from_predecessor_matrix', methods=['POST'])
def draw_graph_from_predecessor_matrix_route():
    data = request.get_json()
    predecessor_matrix = data.get('matrix', [])

    result = draw_graph_from_predecessor_matrix(predecessor_matrix)

    return jsonify(result)

def is_symmetric(matrix):
    # Vérifier que la matrice est une liste bidimensionnelle
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        return False

    # Vérifier que toutes les lignes de la matrice ont la même longueur
    num_rows = len(matrix)
    if not all(len(row) == num_rows for row in matrix):
        return False

    # Vérifier la symétrie de la matrice
    return all(matrix[i][j] == matrix[j][i] for i in range(num_rows) for j in range(num_rows))






@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    data = request.get_json()
    matrix = data.get('matrix', [])

    if not matrix or not matrix[0]:
        return jsonify({'error': "La matrice est vide ou mal formée."})

    print("Dimensions de la matrice:", len(matrix), "x", len(matrix[0]))
    print("Matrice :", matrix)

    # Vérifiez la symétrie de la matrice
    is_symmetric_matrix = is_symmetric(matrix)

    # Utilisez la matrice pour générer le graphe
    G = nx.DiGraph() if not is_symmetric_matrix else nx.Graph()

    for i in range(len(matrix)):
        if i >= len(matrix) or not matrix[i]:
            print(f"Skipping row {i} because it's out of range or empty.")
            continue

        for j in range(len(matrix[i])):
            if j >= len(matrix[i]):
                print(f"Skipping column {j} in row {i} because it's out of range.")
                continue

            if matrix[i][j] != 0:
                # Ajoutez l'orientation et le poids si le graphe est orienté
                if not is_symmetric_matrix:
                    G.add_edge(i, j, weight=matrix[i][j])
                else:
                    G.add_edge(i, j, weight=matrix[i][j])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=10, connectionstyle="arc3,rad=0.1")

    # Sauvegarder la figure dans une mémoire tampon
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convertir la figure en base64 pour l'afficher dans l'HTML
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Récupérer les orientations et les poids des arêtes
    orientations = []
    for edge in G.edges(data=True):
        orientations.append(([pos[edge[0]][0], pos[edge[0]][1]], [pos[edge[1]][0], pos[edge[1]][1]], edge[2]['weight']))

    return jsonify({'image_data': data, 'orientations': orientations, 'weights': matrix})




from bs4 import BeautifulSoup


# ... (code Flask existant) ...

import networkx as nx
import numpy as np
from numpy import inf


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def BellmanFord(self, src):
        dist = [float("Inf")] * self.V
        pred = [-1] * self.V
        dist[src] = 0

        for _ in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u

        for u, v, w in self.graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                return "Graph contains negative weight cycle"

        return dist, pred


    def get_graph(self, src):
        G = nx.DiGraph()
        for u, v, w in self.graph:
            G.add_edge(u, v, weight=w)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Graph Visualization")

        # Save the plot to BytesIO
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Convert the image to base64
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        return img_base64


    def get_bellman_graph(self, src):
        dist, pred = self.BellmanFord(src)

        G = nx.DiGraph()
        for u, v, w in self.graph:
            G.add_edge(u, v, weight=w)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        # Highlight the path from Bellman-Ford results
        path_edges = [(pred[v], v) for v in range(len(pred)) if pred[v] != -1]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)

        plt.title("Bellman-Ford Graph Visualization")

        # Save the plot to BytesIO
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Convert the image to base64
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        
        return img_base64







@app.route('/AppliquerBellman', methods=['POST'])
def index():
    data = request.get_json()
    print("Données JSON reçues :", data.get("requestData"))

    # Extraire la matrice du JSON
    matrix = data.get('requestData', {}).get('matrix', [])
    start_node = data.get('requestData', {}).get('start_node')

    print("La matrice est ", matrix)
    print("Le nœud de départ est ", start_node)

    # Créez un objet Graph
    g = Graph(len(matrix))

    # Ajoutez les arêtes à partir de la matrice
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            weight = matrix[i][j]
            if weight != 0:
                g.addEdge(i, j, weight)

    # Obtenez la réponse du Bellman-Ford
    response = g.BellmanFord(int(start_node))

    if isinstance(response, str):  # Si la réponse est une chaîne (message d'erreur)
        return jsonify({'error': response})
    else:
        dist, pred = response
        image_base64 = [g.get_bellman_graph(int(start_node))]  # Changez l'indice source au besoin
        indices = range(len(image_base64))
        return jsonify({'image_data': image_base64, 'dist': dist, 'pred': pred, 'error': None})


    # return render_template('in.html', image_base64=image_base64, indices=indices)




if __name__ == '__main__':
    app.run(debug=True)






