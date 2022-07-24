import numpy as np
import pandas as pd
import json


def get_the_tokens(file_path):
    """
    A general function which will import the user sessions and will create the X and Y data.
    X are the user sequences serving as input for the model.
    Y are the target values.
    Also returns a unique items dictionary with {item_id : number} pairs. It is always created in the same way,
    to keep it consistent and avoid training labeling models.
    :param file_path: The path where the item is held as string.
    :return: X, Y, and the unique item dictionary.
    """
    # create an empty dictionary to save the sequences from the file
    df = {}
    with open(file_path, 'r') as the_file:
        while True:
            line = the_file.readline()
            # when the looping reaches the end of the file, it stops
            if not line:
                break
            # the data are in JSON format - every line is as JSON
            # apply lamda to decode the line
            line = (lambda x: json.loads(x))(line)
            # add to the empty dictionary the {sequence_ID : [[user_sequence], target]
            df.update({
                (lambda x: list(x.keys())[0])(line): [(lambda x: list(x.values())[0])(line)[:-1],
                                                      (lambda x: list(x.values())[0])(line)[-1]]
            })
    # create a dataframe from the dictionary
    df = pd.DataFrame.from_dict(df, orient='index')
    # fix the names and column types
    df.rename(columns={0: 'sequence', 1: 'target'}, inplace=True)
    df['target'] = df['target'].astype(int)
    # import clusters and create leaders
    clustered_df = pd.read_csv('clustered_items_by_max_appearance_ver2.csv')
    # create dictionary {item id : cluster label}
    mini_df = clustered_df.set_index('item_id')['labels'].to_dict()
    # create dictionary {cluster label: max item id}
    mini_df_2 = clustered_df.sort_values(by='date', ascending=False). \
        drop_duplicates(['labels'], keep='first').set_index('labels')['item_id'].to_dict()
    # map the target values to group representatives
    df['target'] = df['target'].map(mini_df)
    df['target'] = df['target'].map(mini_df_2)
    # get the X values
    X = df['sequence'].values.tolist()
    # get the Y values
    Y = df['target'].values.tolist()
    # create a unique items dictionary
    unique_items_dict = {}
    unique_items_to_pred = set(sorted(Y))
    for i, items in enumerate(unique_items_to_pred):
        unique_items_dict.update(
            {items: i}
        )
    # return X, Y, and the unique items dictionary
    return X, Y, unique_items_dict


def encoded_dataframe(saved_item_vectors_csv_path):
    """
    The algorithm to encode the categorical items. The Frequency Encoder.
    As input it needs the vectors of the items.
    :param saved_item_vectors_csv_path: The string holding the path of the file with the features.
    :return: The encoded dataset.
    """
    # Import the dataset
    dataset = pd.read_csv(saved_item_vectors_csv_path, index_col=0)
    # Fill empty with 0
    dataset.fillna(0, inplace=True)
    for j in dataset.columns:
        dataset[j] = dataset[j].apply(str)
    dataset.sort_index(inplace=True)
    # encode the categorical data
    customEncoder = {}
    for j in dataset.columns:
        customEncoder[j] = (np.round(dataset[j].value_counts() / len(dataset), decimals=7)).to_dict()
    dataset.replace(customEncoder, inplace=True)
    # return the encoded dataset
    return dataset
