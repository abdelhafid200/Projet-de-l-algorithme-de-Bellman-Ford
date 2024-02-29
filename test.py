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


# Python3 program for Bellman-Ford's single source
# shortest path algorithm.

# Class to represent a graph
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
                print("Graph contains negative weight cycle")
                return

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

    

@app.route('/')
def index():
    g = Graph(5)
    g.addEdge(0, 1, 1)  # Adjust indices
    g.addEdge(0, 3, 5)
    g.addEdge(2, 4, 2)
    g.addEdge(3, 1, 3)
    g.addEdge(3, 4, 3)
    g.addEdge(4, 1, 6)
    g.addEdge(1, 2, 1)

    image_base64 = [g.get_bellman_graph(0)]  # Change the indices as needed
    indices = range(len(image_base64))

    return render_template('in.html', image_base64=image_base64, indices=indices)



if __name__ == '__main__':
    app.run(debug=True)
