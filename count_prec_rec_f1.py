import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm
from keras import models
import heapq
from capstone_utilities import get_the_tokens, encoded_dataframe


# define the path of the file with the user sessions
file_path = 'user_sessions.txt'
# import the dataset with the item vectors
items_vectors = 'item_vectors_summed.csv'
dataset = encoded_dataframe(items_vectors)
# import candidate items
cadidate_dataset = pd.read_csv('candidate_items.csv')
candidate_items_list = cadidate_dataset['item_id'].values.tolist()
relevant_items_number = len(candidate_items_list)
# encode the imported data
encoded_items_vect_dictionary = {}
for some_index in list(dataset.index):
    encoded_items_vect_dictionary.update({
        some_index: dataset.loc[some_index].values.tolist()
    })
# assign path to save training graphs
path_to_save_graphs = 'training_data_results_summed'
# split the data
train_values, target_values, labelled_items = get_the_tokens(file_path)
# complete preprocessing
for k in range(len(train_values)):
    for extra_k in range(len(train_values[k])):
        train_values[k][extra_k] = np.array(encoded_items_vect_dictionary[eval(train_values[k][extra_k])])
    train_values[k] = np.mean(train_values[k], axis=0)
    target_values[k] = labelled_items[target_values[k]]
# cast train and target to arrays
train_values = np.array(train_values)
target_values = np.array(target_values)
# import the model
predicting_model = models.load_model('rnn_with_features_classifier_ver2.h5')
# create a file to save the data
f = open('classification_metrics_ver2.csv', 'w')
f.write('precision,recall,f1_score\n')
len_of_data = len(target_values)
# complete testing in batches of 500
for mini_batch in tqdm(range(0, len_of_data, 500)):
    # get the data and the actual targets
    array_of_predictions = predicting_model.predict(train_values[mini_batch:mini_batch+500])
    true_targets = target_values[mini_batch:mini_batch+500]
    for j in range(len(array_of_predictions)):
        # find the indices of top 10
        top_10_indices = heapq.nlargest(10, range(len(array_of_predictions[j])), array_of_predictions[j].__getitem__)
        # get the keys of indices
        keys_of_indices = [k for k, v in labelled_items.items() if v in top_10_indices]
        # calculate precision and recall
        true_positives = 0
        relevants = 0
        for predicted_item in keys_of_indices:
            if predicted_item in candidate_items_list:
                try:
                    true_positives += 1
                    similarity_cos = 1 - cosine(dataset.loc[predicted_item], dataset.loc[true_targets[j]])
                    if similarity_cos >= 0.80:
                        relevants += 1
                except KeyError:
                    continue
        precision = true_positives/10
        try:
            recall = round(relevants/true_positives, ndigits=2)
        except ZeroDivisionError:
            recall = 1
        f1_score = round((2 * precision * recall)/(precision + recall), ndigits=2)
        f.write(str(precision)+','+str(recall)+','+str(f1_score)+'\n')
f.close()
