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
nx.set_node_attributes(G0, False, name="purchase_70")
nx.set_node_attributes(G0, False, name="purchase_150")
nx.set_node_attributes(G0, False, name="purchase_989")
nx.set_node_attributes(G0, False, name="purchase_16326")
#G0.nodes[145]['purchase'] = True
#print(G0.nodes[31383]['purchase'])


# Setting node attributes number of plays for every artist: 'listened_ID'
artist_70 = artist_df[' artistID'] == 70
artist_150 = artist_df[' artistID'] == 150
artist_989 = artist_df[' artistID'] == 989
artist_16326 = artist_df[' artistID'] == 16326

artist_70_df = artist_df[artist_70]
artist_150_df = artist_df[artist_150]
artist_989_df = artist_df[artist_989]
artist_16326_df = artist_df[artist_16326]

nx.set_node_attributes(G0, 0, name="listened_70")
nx.set_node_attributes(G0, 0, name="listened_150")
nx.set_node_attributes(G0, 0, name="listened_989")
nx.set_node_attributes(G0, 0, name="listened_16326")

for node in G0.nodes():
    if node in list(artist_70_df.loc[:, 'userID']):
        G0.nodes[node]['listened_70'] = artist_70_df.loc[int((artist_70_df['userID'] == node).index.tolist()[0]), '#plays']

    if node in list(artist_150_df.loc[:, 'userID']):
        G0.nodes[node]['listened_150'] = artist_150_df.loc[int((artist_150_df['userID'] == node).index.tolist()[0]), '#plays']

    if node in list(artist_989_df.loc[:, 'userID']):
        G0.nodes[node]['listened_989'] = artist_989_df.loc[int((artist_989_df['userID'] == node).index.tolist()[0]), '#plays']

    if node in list(artist_16326_df.loc[:, 'userID']):
        G0.nodes[node]['listened_16326'] = artist_16326_df.loc[int((artist_16326_df['userID'] == node).index.tolist()[0]), '#plays']


# Analyze the edge creation probability
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
        probability_list[i] = sum_real_hist[i]/sum_common_hist[i]
    else:
        probability_list[i] = 0

print(probability_list)

# Simulation until t=6
Graph = G0.copy()
for i in range(1, 7):
    print('i =', i)
    Graph_prev = Graph.copy()
    # Building the new Graph
    for user1 in nx.nodes(Graph):
        for user2 in nx.nodes(Graph):
            if user1 != user2 and Graph.has_edge(user1, user2) is False:
                sum_common = len(list(nx.common_neighbors(Graph, user1, user2)))
                if random.random() < probability_list[sum_common]:
                    Graph.add_edge(user1, user2)

    # Calculating the purchase probabilities and update purchase per artist for every node
    for user in nx.nodes(Graph):
        Bt_70 = 0
        Bt_150 = 0
        Bt_989 = 0
        Bt_16326 = 0
        Nt = len(list(nx.neighbors(Graph, user)))
        for neighbor in nx.neighbors(Graph, user):
            if Graph_prev.nodes[neighbor]['purchase_70']:
                Bt_70 += 1
            if Graph_prev.nodes[neighbor]['purchase_150']:
                Bt_150 += 1
            if Graph_prev.nodes[neighbor]['purchase_989']:
                Bt_989 += 1
            if Graph_prev.nodes[neighbor]['purchase_16326']:
                Bt_16326 += 1

        # For artist 70
        if Graph.nodes[user]['listened_70'] != 0:
            purchase_prob_70 = (Graph.nodes[user]['listened_70'] * Bt_70) / (1000 * Nt)
        else:
            purchase_prob_70 = Bt_70 / Nt
        if random.random() < purchase_prob_70:
            Graph.nodes[user]['purchase_70'] = True

        # For artist 150
        if Graph.nodes[user]['listened_150'] != 0:
            purchase_prob_150 = (Graph.nodes[user]['listened_150'] * Bt_150) / (1000 * Nt)
        else:
            purchase_prob_150 = Bt_150 / Nt
        if random.random() < purchase_prob_150:
            Graph.nodes[user]['purchase_150'] = True

        # For artist 989
        if Graph.nodes[user]['listened_989'] != 0:
            purchase_prob_989 = (Graph.nodes[user]['listened_989'] * Bt_989) / (1000 * Nt)
        else:
            purchase_prob_989 = Bt_989 / Nt
        if random.random() < purchase_prob_989:
            Graph.nodes[user]['purchase_989'] = True

        # For artist 16326
        if Graph.nodes[user]['listened_16326'] != 0:
            purchase_prob_16326 = (Graph.nodes[user]['listened_16326'] * Bt_16326) / (1000 * Nt)
        else:
            purchase_prob_16326 = Bt_16326 / Nt
        if random.random() < purchase_prob_16326:
            Graph.nodes[user]['purchase_16326'] = True

    print('end i =', i)



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






