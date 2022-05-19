import numpy
import pandas as pd
import numpy as np
import networkx as nx
import random


# Building the new Graph
def build_graph(graph_prev):
    graph = graph_prev.copy()
    for user1 in nx.nodes(graph_prev):
        for user2 in nx.nodes(graph_prev):
            if user1 != user2 and (user2 not in nx.neighbors(graph_prev, user1)):
                sum_common = len(list(nx.common_neighbors(graph_prev, user1, user2)))
                if random.random() < probability_list[sum_common]:
                    graph.add_edge(user1, user2)
    return graph


# Running the spread of node group
def purchase_simulating_per_time(Graph_prev, node_list):
    infected = node_list[:]
    may_be_infected = []
    for node in infected:
        may_be_infected += nx.neighbors(Graph_prev, node)
        may_be_infected = list(set(may_be_infected))

    for may_be in may_be_infected:
        if may_be in infected:
            continue
        neighbors = list(nx.neighbors(Graph_prev, may_be))
        Nt = len(neighbors)
        Bt = len(numpy.intersect1d(neighbors, infected))

        if Graph_prev.nodes[may_be]['listened'] != 0:
            purchase_prob = (Graph_prev.nodes[may_be]['listened'] * Bt) / (1000 * Nt)
        else:
            purchase_prob = Bt / Nt

        if random.random() <= purchase_prob:
            infected += [may_be]

    return infected


# simulating the spread that each group of nodes can cause for 6 times
def compute_IC(node_list):
    print("start IC")
    spread = 0
    for i in range(30):
        print("IC ", i)
        infected1 = purchase_simulating_per_time(G0, node_list)
        infected2 = purchase_simulating_per_time(G1, infected1)
        infected3 = purchase_simulating_per_time(G2, infected2)
        infected4 = purchase_simulating_per_time(G3, infected3)
        infected5 = purchase_simulating_per_time(G4, infected4)
        infected6 = purchase_simulating_per_time(G5, infected5)
        spread += len(infected6)

    return spread / 30


# finding the 5 influencers using hill climb method
def hill_climb(shrink_nodes):
    influencers = []
    for i in range(5):
        print("hill_climb ", i)
        best_influencer = -1
        best_spread = -np.inf
        nodes = set(shrink_nodes) - set(influencers)
        for node in nodes:
            spread = compute_IC(influencers + [node])
            if spread > best_spread:
                best_spread = spread
                best_influencer = node

        influencers.append(best_influencer)
    return influencers


# Calculating the purchase probabilities and update purchase per artist for every node
def purchase_probability(graph, graph_prev, infected):
    for node in graph.nodes():
        Nt = len(list(nx.neighbors(graph, node)))
        neighbors = list(nx.neighbors(graph_prev, node))
        Bt = len(numpy.intersect1d(neighbors, infected))

        if graph.nodes[node]['listened'] != 0:
            purchase_prob = (graph.nodes[node]['listened'] * Bt) / (1000 * Nt)
        else:
            purchase_prob = Bt / Nt
        if random.random() <= purchase_prob:
            infected += [node]
            infected = list(set(infected))
    return infected


####################################################################################################
instaglam0_df = pd.read_csv('instaglam0.csv')
instaglam_1_df = pd.read_csv('instaglam_1.csv')
artist_df = pd.read_csv('spotifly.csv')

# Build the graphs
G0 = nx.from_pandas_edgelist(instaglam0_df, 'userID', 'friendID')
G_1 = nx.from_pandas_edgelist(instaglam_1_df, 'userID', 'friendID')

# Set node attributes: 'purchase'=True/False
#nx.set_node_attributes(G0, False, name="purchase")

# Filtering the dataframe for the relevant artist
artist = artist_df[' artistID'] == 16326
artist_df = artist_df[artist]

# Setting number plays of artist for every node
nx.set_node_attributes(G0, 0, name="listened")
for node in G0.nodes():
    if node in list(artist_df.loc[:, 'userID']):
        G0.nodes[node]['listened'] = artist_df.loc[int((artist_df['userID'] == node).index.tolist()[0]), '#plays']

# Analyze the edge creation probability
sum_common_hist = [0] * (G_1.number_of_nodes() - 2)
sum_real_hist = [0] * (G_1.number_of_nodes() - 2)
for user1 in nx.nodes(G_1):
    for user2 in nx.nodes(G_1):
        if user1 != user2 and (user2 not in nx.neighbors(G_1, user1)):
            sum_common = len(list(nx.common_neighbors(G_1, user1, user2)))
            sum_common_hist[sum_common] += 1
            if G0.has_edge(user1, user2):
                sum_real_hist[sum_common] += 1

probability_list = [0] * (G_1.number_of_nodes() - 2)
for i in range(len(sum_common_hist)):
    if sum_common_hist[i] != 0:
        probability_list[i] = sum_real_hist[i] / sum_common_hist[i]
    else:
        probability_list[i] = 0
    if i >= 19:
        probability_list[i] = max(probability_list)

# Building the new graphs for 6 period of time
print("start building graph", 1)
G1 = build_graph(G0)
print("start building graph", 2)
G2 = build_graph(G1)
print("start building graph", 3)
G3 = build_graph(G2)
print("start building graph", 4)
G4 = build_graph(G3)
print("start building graph", 5)
G5 = build_graph(G4)
print("start building graph", 6)
G6 = build_graph(G5)

# narrow the number of node that are candidate to be influencers
nx.set_node_attributes(G0, 0, name="sum_listened")
largest_cc = max(nx.connected_components(G0), key=len)
max_neighbors = 0

for node in G0.nodes():
    if node in largest_cc:
        if len(list(nx.neighbors(G0, node))) > max_neighbors:
            max_neighbors = len(list(nx.neighbors(G0, node)))

shrink_graph_nodes = []
for node in G0.nodes():
    if node in largest_cc:
        G0.nodes[node]['grade_neighbors'] = len(list(nx.neighbors(G0, node))) / max_neighbors
        for neighbor in nx.neighbors(G0, node):
            G0.nodes[node]['sum_listened'] += G0.nodes[neighbor]['listened']
        G0.nodes[node]['Grade'] = G0.nodes[node]['sum_listened'] * G0.nodes[node]['grade_neighbors']
        if G0.nodes[node]['Grade'] > 3.609375:
            shrink_graph_nodes.append(node)


# calling to hill climb algorithm function
purchased = []
best_influencers = hill_climb(shrink_graph_nodes)
print("best influencers:", best_influencers)
for influencer in best_influencers:
    purchased += [influencer]
    #G0.nodes[influencer]['purchase'] = True

# Checking the purchases throw 6 times
purchased = purchase_probability(G1, G0, purchased)
purchased = purchase_probability(G2, G1, purchased)
purchased = purchase_probability(G3, G2, purchased)
purchased = purchase_probability(G4, G3, purchased)
purchased = purchase_probability(G5, G4, purchased)
purchased = purchase_probability(G6, G5, purchased)

print('count =', len(set(purchased)))
#print('influencers =', best_influencers)









































"""
new_infected = node_list[:]
# for each newly infected nodes, find its neighbors that becomes infected
while new_infected:
    infected_nodes = []
    for node in new_infected:

        for neighbor in nx.neighbors(Graph_prev, node):
            neighbors = list(nx.neighbors(Graph_prev, neighbor))
            Nt = len(neighbors)
            Bt_16326 = len(numpy.intersect1d(neighbors, infected))

            if Graph_prev.nodes[neighbor]['listened_16326'] != 0:
                purchase_prob_16326 = (Graph_prev.nodes[neighbor]['listened_16326'] * Bt_16326) / (1000 * Nt)
            else:
                purchase_prob_16326 = Bt_16326 / Nt

            if random.random() <= purchase_prob_16326:
                infected_nodes += [neighbor]

    new_infected = list(set(infected_nodes) - set(infected))
    infected += new_infected

return infected
"""