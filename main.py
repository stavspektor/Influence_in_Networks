import pandas as pd
import numpy as np
import networkx as nx
import random

instaglam0_df = pd.read_csv('instaglam0.csv')
instaglam_1_df = pd.read_csv('instaglam_1.csv')
artist_df = pd.read_csv('spotifly.csv')

# Build the graphs
G0 = nx.from_pandas_edgelist(instaglam0_df, 'userID', 'friendID')
G_1 = nx.from_pandas_edgelist(instaglam_1_df, 'userID', 'friendID')
"""
# Build the new edges graph
Diff = nx.difference(G0, G_1)
"""

# set node attributes: 'purchase'=True/False
nx.set_node_attributes(G0, False, name="purchase")
#G0.nodes[145]['purchase'] = True
#print(G0.nodes[31383]['purchase'])


# setting node attributes: 'listened'=True/False
our_artists = (artist_df[' artistID'] == 70) | (artist_df[' artistID'] == 150) | (artist_df[' artistID'] == 989) | (
            artist_df[' artistID'] == 16326)
our_artists_df = artist_df[our_artists]
nx.set_node_attributes(G0, False, name="listened")
for node in G0.node():
    if node in our_artists_df['userID']:
        G0.nodes[node]['listened'] = True
########## maybe 4 attributes of listen one for every artist
########## and 4 filters


# analyze the edge creation probability
sum_common_hist = [0] * (G_1.number_of_nodes() - 2)
sum_real_hist = [0] * (G_1.number_of_nodes() - 2)
for user1 in nx.nodes(G_1):
    for user2 in nx.nodes(G_1):
        if user1 != user2 and G_1.has_edge(user1,user2) is False:
            sum_common = len(list(nx.common_neighbors(G_1, user1, user2)))
            sum_common_hist[sum_common] += 1
            if G0.has_edge(user1,user2) is True:
                sum_real_hist[sum_common] += 1

probability_list = [0] * (G_1.number_of_nodes() - 2)
for i in range(len(sum_common_hist)):
    if sum_common_hist[i] != 0:
        probability_list[i] = sum_real_hist[i]/sum_common_hist[i]
    else:
        probability_list[i] = 0


# Simulation until t=6
Graph = G0
Graph_prev = G0
for i in range(6):
    Graph_prev = Graph
    for user1 in nx.nodes(Graph):
        for user2 in nx.nodes(Graph):
            if user1 != user2 and Graph.has_edge(user1, user2) is False:
                sum_common = len(list(nx.common_neighbors(Graph, user1, user2)))
                if random.random() < probability_list[sum_common]:
                    Graph.add_edge(user1, user2)





#filter = new.groupby('userID').size() == 2










"""
import matplotlib.pyplot as plt
nx.draw(Graph, with_labels=True)
plt.show()
"""


"""
# Check triadic closure
triadic = []
for user1 in nx.nodes(G_1):
    for user2 in nx.nodes(G_1):
        if user1 != user2:
            common_neighbors_list = list(nx.common_neighbors(G_1, user1, user2))
            if len(common_neighbors_list) >= 1:
                if (user1, user2) not in G_1.edges():
                    triadic.append((user1, user2, len(common_neighbors_list)))


# create DF and graph for the triadic
list_user1 = []
list_user2 = []
for i in range(len(triadic)):
    for j in range(triadic[i][2]):
        list_user1.append(triadic[i][0])
        list_user2.append(triadic[i][1])

predicted_edges = {'userID': list_user1, 'friendID': list_user2}
predicted_edges_df = pd.DataFrame(predicted_edges)
PE = nx.from_pandas_edgelist(predicted_edges_df, 'userID', 'friendID')


# proportion of the diff edges to the triadic prediction
probability = Diff.number_of_edges() / PE.number_of_edges()
print(probability)



# Simulating Pareto distribution
x_min = 1
n = G_1.number_of_nodes()
sum_n = 0
for node in nx.nodes(G_1):
    sum_n += np.log(len(list(nx.neighbors(G_1, node))) / x_min)
alpha = 1 + n * (1/sum_n)
#drawing samples from distribution
samples = (np.random.pareto(alpha, 1000)+1) * x_min
print(alpha)

"""






