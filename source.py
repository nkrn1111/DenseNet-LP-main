import os

'''parameters from gui'''
Graph_name=""
Graph_absoloute_path=""
version="v1.0.0"
progress_cnt = 1


K = 64
DimensionOfFeatures = 64
distance = 3
batch_size = 32
percentage_of_edges_to_remove = 0.1
percentage_of_negative_class_additions = 1 - percentage_of_edges_to_remove

''' Menu params'''
GitHub = 'https://github.com/dimaxer/DenseNet-LP'
GishaText_9="-family {Gisha} -size 9"
about_header_instructions = "Link Prediction Project-Ort Braude\n version : "+ version + "\nNoam Keren\tDimitry yavestigneyev"
about_footer_instructions = ""


'''path to the data, both where to save the data and from where to read the data '''
data_path = "data"
# path to save results csv and the model
results_path = "results"
# path for the graph file, labels ,features_vectors , and data_sets, number_of_nodes 
graph_path = "graph"
labels_path = os.path.join(data_path, "labels")
feature_vectors_path = os.path.join(data_path, "feature_vectors")
data_set_path = os.path.join(data_path, "data_set")



'''file names'''
training_set_node2vec_feature_vector_file_name = "training_set_features_vector.txt"
test_set_node2vec_feature_vector_file_name = "test_set_features_vector.txt"

training_data_file_name = "train_data_file.npy"
train_labels_file_name = "train_labels.npy"

test_data_file_name = "test_data_file.npy"
test_labels_file_name = "test_labels.npy"

graph_file_name = "USAir.txt"

num_of_nodes_file_name = "number_of_nodes.txt"

'''merge filenames with the path'''
training_set_node2vec_feature_vector_file_name = os.path.join(feature_vectors_path, training_set_node2vec_feature_vector_file_name)
test_set_node2vec_feature_vector_file_name = os.path.join(feature_vectors_path, test_set_node2vec_feature_vector_file_name)
training_data_file_name = os.path.join(data_set_path, training_data_file_name)
test_data_file_name = os.path.join(data_set_path, test_data_file_name)
train_labels_file_name = os.path.join(labels_path, train_labels_file_name)
test_labels_file_name = os.path.join(labels_path, test_labels_file_name)
graph_file_name = os.path.join(graph_path, graph_file_name)
num_of_nodes_file_name = os.path.join(data_path, num_of_nodes_file_name)

