import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import random

from networkx import graph_edit_distance

instaglam0_df = pd.read_csv('instaglam0.csv')
instaglam_1_df = pd.read_csv('instaglam_1.csv')
artist_df = pd.read_csv('spotifly.csv')


# Build the graphs
G0 = nx.from_pandas_edgelist(instaglam0_df, 'userID', 'friendID')
G_1 = nx.from_pandas_edgelist(instaglam_1_df, 'userID', 'friendID')

nx.set_node_attributes(G0, 0, name="listened_70")

sum_common_hist = [0] * (G_1.number_of_nodes() - 2)
sum_real_hist = [0] * (G_1.number_of_nodes() - 2)
for user1 in nx.nodes(G_1):
    for user2 in nx.nodes(G_1):
        if user1 != user2 and G_1.has_edge(user1, user2) is False:
            sum_common = len(list(nx.common_neighbors(G_1, user1, user2)))
            sum_common_hist[sum_common] += 1
            if G0.has_edge(user1, user2) is True:
                sum_real_hist[sum_common] += 1
probability_list = [0] * (G_1.number_of_nodes() - 2)
for i in range(len(sum_common_hist)):
    if sum_common_hist[i] != 0:
        if i != 0:
            probability_list[i] = sum_real_hist[i] / (sum_common_hist[i])
        else:
            probability_list[i] = sum_real_hist[i] / sum_common_hist[i]
    else:
        probability_list[i] = 0

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
z = np.polyfit(np.log(x), np.log(probability_list), 1)
plt.plot(z)





"""
# Building the new Graph
Graph = G_1.copy()
for user1 in nx.nodes(Graph):
    for user2 in nx.nodes(Graph):
        if user1 != user2 and Graph.has_edge(user1, user2) is False:
            sum_common = len(list(nx.common_neighbors(Graph, user1, user2)))
            if random.random() < probability_list[sum_common]:
                Graph.add_edge(user1, user2)

Diff = nx.difference(G0, Graph)
Diff2 = nx.difference(Graph, G0)
"""