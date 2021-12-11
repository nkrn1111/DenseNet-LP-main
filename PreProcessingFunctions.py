from networkx.classes.function import number_of_nodes

import source
from Gui import unknown_support, unknown
from source import *
from numba import jit, cuda
import networkx as nx
import math
from node2vec import Node2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import itertools as it
import random
from tqdm import tqdm
import warnings


# import sys
# np.set_printoptions(threshold=sys.maxsize)

def create_node2vec_feature_vectors_to_file(graph, file_name):
    print("Create and save node2vec feature vectors into file:", file_name)
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=DimensionOfFeatures, walk_length=30, num_walks=200,
                        workers=4)  # Use temp_folder for big graphs
    print("Embedding please wait for the process to finish")
    # Embed nodes
    model = node2vec.fit(window=10, min_count=1,
                         batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Save embeddings for later use
    model.wv.save_word2vec_format(file_name)


# extract a subgraph for a specific node which all subsequesnt node are <= to that distance
def subgraph_extraction(min_dist, length_dictionary):
    size = len(length_dictionary)
    list_of_min_dist_list = []
    for i in range(0, size):
        min_dist_list = []
        for j in range(0, size):
            # print("length between ", i, " and ", j, " is: ", dictionary[i][j])
            try:
                if length_dictionary[i][j] <= min_dist and i != j:
                    min_dist_list.append(j)
            except KeyError:
                k = 3
        # print("min_dist_list at ", i, " :", min_dist_list)

        list_of_min_dist_list.append(min_dist_list)
    return list_of_min_dist_list


# parse string to int
def parsing_int(row):
    index = 0
    node_string = ""
    while row[index] != ' ':
        node_string += row[index]
        index = index + 1
    node = int(node_string)
    return node, index + 1


# parse string to double
def parsing_double(row, index):
    float_string = ""
    try:
        while row[index] != ' ' or index == len(row):
            float_string += row[index]
            index = index + 1
    except IndexError:
        k = 5
    return float(float_string), index + 1


# return a feature matrix by reading the values from a text file
def txt_file_to_matrix(file_name, number_of_nodes):
    print("Save feature matrix into text file:", file_name)
    feature_vector_matrix = np.zeros((number_of_nodes, DimensionOfFeatures))
    with open(file_name) as FILE:
        first_line = 0
        for vector in FILE:
            if first_line == 0:
                first_line = 1
            else:
                node_number, row_index = parsing_int(vector)
                # TODO: FIXME work around for USAir dataset because node numbers start from 1 instead of 0
                node_number = node_number - 1
                for i in range(DimensionOfFeatures):
                    feature_vector_matrix[node_number][i], row_index = parsing_double(vector, row_index)
        FILE.close()
    return feature_vector_matrix


# create a feature vecor matrix for the subgraph for a specific node
def subgraph_feature_vector_matrix(node_number, feature_matrix, k, distance_sub_graph, distance_dictionary):
    sub_graph_feature_matrix = np.zeros((k, DimensionOfFeatures))
    subgraph_for_all_nodes = subgraph_extraction(distance_sub_graph, distance_dictionary)
    subgraph_list_for_node = subgraph_for_all_nodes[node_number]
    for i in range(k):
        for j in range(DimensionOfFeatures):
            if i < len(subgraph_list_for_node):
                sub_graph_feature_matrix[i][j] = feature_matrix[subgraph_list_for_node[i]][j]
    return sub_graph_feature_matrix


# Calculate the cosine distance of the sub graph feature vector matrix and return a sorted list of the distance and the node number as a tuple
def cosine_distance_similarity(matrix, feature_vector):
    i = 0
    list = []
    for vector in matrix:
        num_array = cosine_similarity(vector.reshape(1, -1), feature_vector.reshape(1, -1))
        list.append((num_array[0][0], i))
        i += 1
    list.sort()

    return list


# return the index from the list of tuples that has cosine distane and node number as a tuple
def list_index(similarity_list, matrix_index):
    index = 0
    for list_tuple in similarity_list:
        if list_tuple[1] == matrix_index:
            return index, list_tuple[0]
        index += 1


# create and return a sorted feature vector matrix
def sorted_similarity_vectors(similarity_list, subGraph_feature_matrix):
    sorted_feature_matrix = np.zeros((K, DimensionOfFeatures))
    matrix_index = 0
    for vector in subGraph_feature_matrix:
        list_tuple = list_index(similarity_list, matrix_index)
        cos_distance = list_tuple[1]
        if cos_distance != 0:
            tuple_index = list_tuple[0]
            sorted_feature_matrix[tuple_index] = vector
        matrix_index += 1
    return sorted_feature_matrix


# create a list of node pairs for example [(node_number1, node_number2),(node_number3, node_number4)]
def generate_node_combinations(num_of_nodes):
    print("Generate node combiantions")
    nodes = list(range(0, num_of_nodes))
    return list(it.combinations(nodes, 2))


# this function creates a (2 x K x K)  matrix for a node pair
def create_node_pair_features_matrix(nodes, feature_matrix, dictionary):
    print("Create pair features matrix")
    node_feature_matrix_pair = np.zeros((2, K, DimensionOfFeatures))
    i = 0
    for node in nodes:
        subGraphFeatureVectorMatrix = subgraph_feature_vector_matrix(node, feature_matrix, K, distance, dictionary)
        cosine_distance_list = cosine_distance_similarity(subGraphFeatureVectorMatrix, feature_matrix[node])
        sorted_matrix = sorted_similarity_vectors(cosine_distance_list, subGraphFeatureVectorMatrix)
        node_feature_matrix_pair[i] = sorted_matrix
        i = i + 1
    return node_feature_matrix_pair


# this function creates a 4 dimensional matrix for all node paris size of: (num_of_node_pairs x 2 x K x K)
def create_all_node_pair_feature_matrices(feature_matrix, node_combination_list, dictionary, num_of_node_pairs):
    print("Create all node pair feature matrices")
    node_pair_feature_matrices = np.zeros((num_of_node_pairs, 2, K, DimensionOfFeatures))
    for i in range(num_of_node_pairs):
        node_pair_feature_matrices[i] = create_node_pair_features_matrix(node_combination_list[i], feature_matrix,
                                                                         dictionary)
    return node_pair_feature_matrices


# this function creates a (3 x K x K)  matrix for a node pair in order to simulate a picture like data. 
# the third channel is the averge between the other 2 channel in the matrix
def convert_node_pair_into_picture_data(nodes, feature_matrix, dictionary):
    node_feature_matrix_pair = np.zeros((3, K, DimensionOfFeatures))
    i = 0
    for node in nodes:
        sub_graph_feature_vector_matrix = subgraph_feature_vector_matrix(node, feature_matrix, K, distance, dictionary)
        cosine_distance_list = cosine_distance_similarity(sub_graph_feature_vector_matrix, feature_matrix[node])
        node_feature_matrix_pair[i] = sorted_similarity_vectors(cosine_distance_list, sub_graph_feature_vector_matrix)

        i = i + 1

    return node_feature_matrix_pair


@cuda.jit
def calcMultiGpu(matrix, num_of_node_pairs):
    for i in range(num_of_node_pairs):
        for l in range(K):
            for p in range(DimensionOfFeatures):
                matrix[i][2][l][p] = (matrix[i][0][l][p] + matrix[i][1][l][p]) / 2.0


# this function creates a 4 dimensional matrix for all node paris (after recreating them as picture like) size of: (num_of_node_pairs x 3 x K x K)
def convert_all_node_pairs_into_picture_data(feature_matrix, node_combination_list, dictionary, num_of_node_pairs):
    print("Convert all node pairs into picture data")
    node_pair_feature_matrices = np.zeros((int(num_of_node_pairs), 3, K, DimensionOfFeatures))
    for i in tqdm(range(num_of_node_pairs)):
        node_pair_feature_matrices[i] = convert_node_pair_into_picture_data(node_combination_list[i], feature_matrix,dictionary)
        unknown.update_progress_bar(num_of_node_pairs * 2,0)
    print(
        "Compute 3rd channel (average of first 2 channels)  for all the node-pairs feature matrix (CNN expects pictures with 3 channels as input)")
    print("Parallel computation using cuda cores beggining please wait...")
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(K / threadsperblock[0])
    blockspergrid_y = math.ceil(K / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calcMultiGpu[blockspergrid, threadsperblock](node_pair_feature_matrices, num_of_node_pairs)
    print("Parallel computation Done!")

    return node_pair_feature_matrices


# this function creates all true lables for all the node pairs. 
# for example: label 0 means a node pair dosent have a link the original graph
# label 1 means a node pair has a link the original graph
def create_pair_labels(node_combination_list, graph):
    print("Create node pair labels")
    label_array = np.zeros(len(node_combination_list), dtype=int)
    index = 0
    for nodePair in node_combination_list:
        if graph.has_edge(nodePair[0], nodePair[1]):
            label_array[index] = 1
        index = index + 1
    return label_array


def save_data_into_file(file_name, final_data):
    print("Save data into file:", file_name)
    np.save(file_name, final_data)


def load_data_from_file(training_file_name):
    return np.load(training_file_name)


def remove_percentage_of_edges_from_graph(graph, percentage_to_remove):
    print("Remove percentage of edges from graph")
    graph_to_return = graph
    removed_edges = random.sample(graph.edges(), k=int(percentage_to_remove * graph.number_of_edges()))
    graph_to_return.remove_edges_from(removed_edges)
    return graph_to_return, removed_edges


def percentage_of_complemant_graph_edges(graph):
    complemant_graph = nx.complement(graph)
    negative_class = complemant_graph.edges()
    negative_class_edges = random.sample(negative_class, k=int(
        (percentage_of_negative_class_additions * graph.number_of_edges()) - (
                percentage_of_edges_to_remove * graph.number_of_edges())))

    return negative_class_edges


def create_data_set_to_file(graph, node2vec_feature_vector_file_name, node_combination_list, data_set_file_name,
                            dictionary, num_of_nodes):
    print("Create data set to file:", data_set_file_name)
    num_of_node_pairs = int((num_of_nodes * (num_of_nodes - 1)) / 2)
    create_node2vec_feature_vectors_to_file(graph, node2vec_feature_vector_file_name)
    # create feature matrix for all the nodes in the graph
    node_pair_feature_matrix = txt_file_to_matrix(node2vec_feature_vector_file_name, num_of_nodes)
    all_node_pairs_feature_matrix = convert_all_node_pairs_into_picture_data(node_pair_feature_matrix,node_combination_list, dictionary,num_of_node_pairs)
    save_data_into_file(data_set_file_name, all_node_pairs_feature_matrix)
    unknown.update_progress_bar(num_of_node_pairs * 2, 1)