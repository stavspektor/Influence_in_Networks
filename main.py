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

# Set node attributes: 'purchase'=True/False
nx.set_node_attributes(G0, False, name="purchase_16326")
#G0.nodes[145]['purchase'] = True
#print(G0.nodes[31383]['purchase'])


# Setting node attributes number of plays for every artist: 'listened_ID'
artist_16326 = artist_df[' artistID'] == 16326
artist_16326_df = artist_df[artist_16326]

nx.set_node_attributes(G0, 0, name="listened_16326")

for node in G0.nodes():
    if node in list(artist_16326_df.loc[:, 'userID']):
        G0.nodes[node]['listened_16326'] = artist_16326_df.loc[int((artist_16326_df['userID'] == node).index.tolist()[0]), '#plays']


# simulating the spread that each group of nodes can cause
def compute_IC(Graph_prev, node_list):
    print("start IC")
    total_spread = 0
    for i in range(10):
        print("IC i =", i)
        infected = node_list[:]
        new_infected = node_list[:]

        # for each newly infected nodes, find its neighbors that becomes infected
        while new_infected:
            infected_nodes = []
            for node in new_infected:
                for neighbor in nx.neighbors(Graph_prev, node):
                    Bt_16326 = 0
                    neighbors = list(nx.neighbors(Graph_prev, neighbor))
                    Nt = len(neighbors)
                    for subneighbor in nx.neighbors(Graph_prev, neighbor):
                        if subneighbor in infected:
                            Bt_16326 += 1

                    if Graph_prev.nodes[neighbor]['listened_16326'] != 0:
                        purchase_prob_16326 = (Graph_prev.nodes[neighbor]['listened_16326'] * Bt_16326) / (1000 * Nt)
                    else:
                        purchase_prob_16326 = Bt_16326 / Nt

                    if random.random() <= purchase_prob_16326:
                        infected_nodes += [neighbor]

            new_infected = list(set(infected_nodes) - set(infected))
            infected += new_infected

        total_spread += len(infected)

    return total_spread / 10


# finding the 5 influencers using hill climb method
def hill_climb(Graph_prev, shrink_nodes):
    influencers = []
    for i in range(5):
        best_influencer = -1
        best_spread = -np.inf
        nodes = set(shrink_nodes) - set(influencers)
        for node in nodes:
            spread = compute_IC(Graph_prev, influencers + [node])
            if spread > best_spread:
                best_spread = spread
                best_influencer = node

        influencers.append(best_influencer)

    return influencers


# Analyze the edge creation probability
sum_common_hist = [0] * (G_1.number_of_nodes() - 2)
sum_real_hist = [0] * (G_1.number_of_nodes() - 2)
for user1 in nx.nodes(G_1):
    for user2 in nx.nodes(G_1):
        if user1 != user2 and (G_1.has_edge(user1, user2) is False):
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


# narrow the number of node that are candidate to be influencers
nx.set_node_attributes(G0, 0, name="sum_listened_16326")
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
            G0.nodes[node]['sum_listened_16326'] += G0.nodes[neighbor]['listened_16326']
        G0.nodes[node]['Grade_16326'] = G0.nodes[node]['sum_listened_16326'] * G0.nodes[node]['grade_neighbors']
        if G0.nodes[node]['Grade_16326'] > 4:
            shrink_graph_nodes.append(node)


# calling to hill climb algorithm function
Graph = G0.copy()
Graph_prev = Graph.copy()
best_influencers = hill_climb(Graph_prev, shrink_graph_nodes)
print("best influencers:", best_influencers)
for influencer in best_influencers:
    Graph_prev.nodes[influencer]['purchase_16326'] = True


count = 0
# Simulation until t=6
for i in range(1, 7):
    print('simulation i =', i)

    # Building the new Graph
    for user1 in nx.nodes(Graph_prev):
        for user2 in nx.nodes(Graph_prev):
            if user1 != user2 and (Graph_prev.has_edge(user1, user2) is False):
                sum_common = len(list(nx.common_neighbors(Graph_prev, user1, user2)))
                if random.random() < probability_list[sum_common]:
                    Graph.add_edge(user1, user2)

    # Calculating the purchase probabilities and update purchase per artist for every node
    for node in Graph.nodes():
        Bt_16326 = 0
        neighbors = list(nx.neighbors(Graph, node))
        Nt = len(neighbors)
        for neighbor in nx.neighbors(Graph_prev, node):
            if Graph_prev.nodes[neighbor]['purchase_16326']:
                Bt_16326 += 1

        if Graph.nodes[node]['listened_16326'] != 0:
            purchase_prob_16326 = (Graph.nodes[node]['listened_16326'] * Bt_16326) / (1000 * Nt)
        else:
            purchase_prob_16326 = Bt_16326 / Nt
        if random.random() <= purchase_prob_16326:
            Graph.nodes[node]['purchase_16326'] = True

    Graph_prev = Graph.copy()


    for node in Graph.nodes():
        if Graph.nodes[node]['purchase_16326']:
            count += 1

    print('count =', count)
print('influencers =', best_influencers)
























"""
count = 0
for node in Graph.nodes():
    if Graph.nodes[node]['purchase_70']:
        count += 1

print('count =', count)
"""



"""
Graded_graph_df = pd.DataFrame.from_dict(dict_Graph)
top_5_70 = Graded_graph_df.nlargest(5, 'Grade_70').tolist()
print(top_5_70)
"""

"""
filter = new.groupby('userID').size() == 2

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






