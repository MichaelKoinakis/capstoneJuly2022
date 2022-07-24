import pandas as pd


# import csv holding the item features
dataset = pd.read_csv('item_features.csv')
# create a pivot table with item_id as index, features_categories as columns and features_values as pivot values
dataset = pd.pivot_table(dataset, index='item_id', columns='feature_category_id', values='feature_value_id',
                         fill_value=0)
# make the pivot having 1 wherever there are values or 0 otherwise
dataset = (dataset/dataset == 1).astype(int)
# create a list of unique items
all_the_indexes = list(dataset.index.values)
# find the number of the unique items
unique_number_of_items = len(all_the_indexes)
print("unique number of items", unique_number_of_items)
# create a list to hold the common teams
teams = []
# create a list to hold the indexes that were used
used_index = []
# loop through the indexes
while len(all_the_indexes):
    # create a list to hold the same items
    selected_indexes = []
    # create an array to test for similarity
    test_array = dataset.loc[all_the_indexes[0]].values
    for j in all_the_indexes:
        # if the tested item does not match the array continue the loop
        if False in (dataset.loc[j].values == test_array):
            continue
        else:
            # if it matches, get the index in one list and remove it from the indexes list
            selected_indexes.append(j)
            all_the_indexes.remove(j)
    print("remaining", len(all_the_indexes))
    teams.append(selected_indexes)
# general check to validate all items are appended in a list
print("created teams", len(teams))
check_correct = 0
for i in teams:
    check_correct += len(i)
print("grouped items", check_correct)
# create a number for each team
labels = {}
for i in range(len(teams)):
    for team_item in teams[i]:
        labels.update({
            team_item: i
        })
# create a column to add the labels
dataset['labels'] = ''
for i in list(dataset.index):
    try:
        dataset['labels'].loc[i] = labels[i]
    except:
        # a general exception in case there was an error before
        dataset['labels'].loc[i] = 99999
# check purchases and group them by item_id
df = pd.read_csv('train_purchases.csv')
df = df[['item_id', 'date']].groupby(by='item_id').count()
df.sort_values(by='date', ascending=False, inplace=True)
# create a join of the purchases dataframe and the items dataframe to get the group number
df = df.join(dataset['labels'])
# save the dataframe to file
df.to_csv('clustered_items_by_max_appearance_ver2.csv')
