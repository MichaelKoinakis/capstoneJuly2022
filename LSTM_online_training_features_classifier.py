import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, LayerNormalization, BatchNormalization
from keras.models import Model
from keras.metrics import SparseTopKCategoricalAccuracy, SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from capstone_utilities import get_the_tokens, encoded_dataframe


# define the path of the file with the user sessions
file_path = 'user_sessions.txt'
# import the dataset with the item vectors
items_vectors = 'item_vectors_summed.csv'
dataset = encoded_dataframe(items_vectors)

encoded_items_vect_dictionary = {}
for some_index in list(dataset.index):
    encoded_items_vect_dictionary.update({
        some_index: dataset.loc[some_index].values.tolist()
    })

# define the DNN model
input_to_model = Input(shape=(39,))
normalization_layer_0 = BatchNormalization()(input_to_model)
dense_layer = Dense(3488, activation='relu')(normalization_layer_0)
dropout_layer_2 = Dropout(0.2)(dense_layer)
normalization_layer = LayerNormalization()(dropout_layer_2)
dense_layer_2 = Dense(3488, activation='relu')(normalization_layer)
dropout_layer_3 = Dropout(0.2)(dense_layer_2)
normalization_layer_2 = LayerNormalization()(dropout_layer_3)
dense_layer_3 = Dense(3488, activation='relu')(normalization_layer_2)
dropout_layer_4 = Dropout(0.2)(dense_layer_3)
normalization_layer_3 = LayerNormalization()(dropout_layer_4)
dense_layer_4 = Dense(872, activation='softmax')(normalization_layer_3)
# compile the model and print the layout
model = Model(inputs=[input_to_model], outputs=dense_layer_4)
# Cosine similarity Loss: -1 = greater, 0 = NaN, 1 = dissimilarity
model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam', metrics=[SparseCategoricalAccuracy(),
                                                                                 SparseTopKCategoricalAccuracy(k=10)])
print(model.summary())

# assign path to save training graphs
path_to_save_graphs = 'training_data_results_summed'
train_values, target_values, labelled_items = get_the_tokens(file_path)

for k in range(len(train_values)):
    for extra_k in range(len(train_values[k])):
        train_values[k][extra_k] = np.array(encoded_items_vect_dictionary[eval(train_values[k][extra_k])])
    train_values[k] = np.mean(train_values[k], axis=0)
    target_values[k] = labelled_items[target_values[k]]

train_values = np.array(train_values)
target_values = np.array(target_values)

print("Number of targets reduced to", len(np.unique(target_values)))
# print(target_values)

train_x, test_x, train_y, test_y = train_test_split(train_values, target_values, test_size=0.2, random_state=42)

# fit the model to new length values
early_stopping = EarlyStopping(monitor='loss', patience=10)
encoder_model = model.fit(x=[train_x], y=train_y, batch_size=128, epochs=1000, callbacks=[early_stopping],
                          validation_split=0.2)
# print(model.evaluate(test_x, test_y))
with open('mlp_classification_top_clustered_ver2.txt', 'w') as another_file:
    another_file.write("Test lost, test accuracy, test_accuracy_at_10:" + str(model.evaluate(test_x, test_y)))
# generate plot for accuracy and export a plot
plt.plot(encoder_model.history['sparse_top_k_categorical_accuracy'], label='Accuracy at 10')
plt.plot(encoder_model.history['sparse_categorical_accuracy'], label='Accuracy')
plt.legend()
plt.title('Model Training Accuracy')
plt.ylabel('Accuracy Level')
plt.xlabel('Epoch')
plt.plot()
plt.savefig(path_to_save_graphs + '/' + 'mae_cosine_DRNN_sequence_len_classifier_ver2.png')
plt.clf()

# save the model to disk
model.save("rnn_with_features_classifier_ver2.h5")
print("Saved model to disk for length")
