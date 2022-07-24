from pyspark import SparkContext
from dateutil import parser
import json
import time
import logging
import warnings
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)
warnings.filterwarnings('ignore')


def write_to_json(some_dictionary):
    """
    Function to write JSON objects to .txt files
    :param some_dictionary: a dictionary with { Key : [Values] } pairs
    :return: Nothing
    """
    json_object = json.dumps(some_dictionary)
    with open('user_sessions.txt', 'a') as outfile:
        outfile.write(json_object)
        outfile.write('\n')


# create an empty txt file
f = open('user_sessions.txt', 'w')
f.close()

# initialize Spark Session
sc = SparkContext("local[1]", 'capstoneUsers')
# create RDDs
rdd_1 = sc.textFile('train_sessions.csv')
rdd_2 = sc.textFile('train_purchases.csv')
# mark the headers to drop them
headers = rdd_1.first()
headers_2 = rdd_2.first()
rdd_2 = rdd_2.filter(lambda row: row != headers_2)
t0 = time.time()
# apply functions to rdd
# first filter the headers. then join the rdds to get their union of session + bought item.
# then split the string. make the dates datetime objects. sort the rdd by the date.
# group the rdd by the key (session name). all interactions are saved in chronological order to list.
# use the function to write to the created .txt file.
rdd_1.filter(lambda row: row != headers).union(rdd_2).map(lambda x: x.split(',')).\
    map(lambda x: [x[0], x[1], parser.parse(x[2])]).sortBy(lambda x: x[2]).map(lambda x: (x[0], x[1])).groupByKey().\
    mapValues(list).map(lambda x: {x[0]: x[1]}).foreach(write_to_json)
t1 = time.time()
print('Elapsed', t1-t0)
