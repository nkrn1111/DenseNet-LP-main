from networkx.classes.function import number_of_nodes
from PreProcessingFunctions import *
import networkx as nx

print("Loading graph from file: ", graph_file_name)

# load graph from text file
# original_graph = nx.fast_gnp_random_graph(n=20, p=0.3)
original_graph = nx.read_edgelist(graph_file_name, create_using= nx.Graph(), nodetype = int)
num_of_nodes = original_graph.number_of_nodes()
negative_class_edges = percentage_of_complemant_graph_edges(original_graph)

#save number of nodes into txt file
f = open(num_of_nodes_file_name, "w")
f.write(str(num_of_nodes))
f.close()


# print loaded graph info
print("Original ",nx.info(original_graph))


# create all node combinations list for all possible node pairs so that the model would be able to ask if there is a link between 2 nodes or not
node_pairs_list = generate_node_combinations(num_of_nodes)
print("Number of node pairs", len(node_pairs_list))


# to simulate link prediction we create a graph without 10% of the edges and graph with only those 10% + 80% negative class edges for the training set and testing set respectively
# create a graph without 10% deleted edges from the orginal graph for the training set
training_set_graph, removed_edges = remove_percentage_of_edges_from_graph(original_graph, percentage_of_edges_to_remove)
print("Training Set Graph after deleting", percentage_of_edges_to_remove,"% of the edges ", nx.info(training_set_graph))


# create a graph with only those 10% deleted edges from the orginal graph for the test set
test_set_graph = nx.Graph()
test_set_graph.add_nodes_from(original_graph)
test_set_graph.add_edges_from(removed_edges)
test_set_graph.add_edges_from(negative_class_edges)
print("Test Set Graph with the deleted", percentage_of_edges_to_remove, "% edges and addition of", (percentage_of_negative_class_additions - percentage_of_edges_to_remove) , "% random negative class edges", nx.info(test_set_graph))


# create graph dictionary for shortest paths
shortest_path_dict_training_set = dict(nx.all_pairs_shortest_path_length(training_set_graph))
shortest_path_dict_test_set = dict(nx.all_pairs_shortest_path_length(test_set_graph))


train_lables = create_pair_labels(node_pairs_list, training_set_graph)
save_data_into_file(train_labels_file_name, train_lables)


test_labels = create_pair_labels(node_pairs_list, test_set_graph)
save_data_into_file(test_labels_file_name, test_labels)


# create training set data files
create_data_set_to_file(training_set_graph, training_set_node2vec_feature_vector_file_name , node_pairs_list, training_data_file_name, shortest_path_dict_training_set, num_of_nodes)


# create test set data files
create_data_set_to_file(test_set_graph, test_set_node2vec_feature_vector_file_name, node_pairs_list, test_data_file_name, shortest_path_dict_test_set, num_of_nodes)


