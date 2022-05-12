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

nx.set_node_attributes(G0, 0, name="listened_16326")
artist_16326 = artist_df[' artistID'] == 16326
artist_16326_df = artist_df[artist_16326]

for node in G0.nodes():
    if node in list(artist_16326_df.loc[:, 'userID']):
        G0.nodes[node]['listened_16326'] = artist_16326_df.loc[int((artist_16326_df['userID'] == node).index.tolist()[0]), '#plays']


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
            probability_list[i] = sum_real_hist[i] / (sum_common_hist[i])
    else:
        probability_list[i] = 0
    if i >= 19:
        probability_list[i] = max(probability_list)




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




largest_cc = max(nx.connected_components(G0), key=len)
max_neighbors = 0

for node in G0.nodes():
    if node in largest_cc:
        if len(list(nx.neighbors(G0, node))) > max_neighbors:
            max_neighbors = len(list(nx.neighbors(G0, node)))

nx.set_node_attributes(G0, 0, name="sum_listened_16326")
toprint=[]
shrink_graph_nodes = []
for node in G0.nodes():
    if node in largest_cc:
        G0.nodes[node]['grade_neighbors'] = len(list(nx.neighbors(G0, node))) / max_neighbors
        for neighbor in nx.neighbors(G0, node):
            G0.nodes[node]['sum_listened_16326'] += G0.nodes[neighbor]['listened_16326']
        G0.nodes[node]['Grade_16326'] = G0.nodes[node]['sum_listened_16326'] * G0.nodes[node]['grade_neighbors']
        if G0.nodes[node]['Grade_16326'] > 2.56359649122807:
            shrink_graph_nodes.append(node)
        #if G0.nodes[node]['Grade_16326'] >= 0.31714595767769943:
        #



    #toprint.append(G0.nodes[node]['Grade_16326'])













