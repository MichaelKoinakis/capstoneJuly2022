from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pandas as pd


# initialize the session
spark = SparkSession.builder.master("local[1]").appName("item_vectors").getOrCreate()
# for windows users - with spark 3.XX appear compatibility issues when trying to show data
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
spark.sql("set spark.sql.debug.maxToStringFields=100")
# create the rdd
df = spark.read.csv('item_features.csv', header=True)
# cast values to integers
df = df.withColumn('feature_value_id', df['feature_value_id'].cast(IntegerType()))
# pivot the dataframe to create the vector for each item
df = df.groupBy('item_id').pivot('feature_category_id').sum('feature_value_id')
# cast the dataframe to pandas
pandas_df = df.toPandas()
percent_missing = pandas_df.isnull().sum() * 100 / len(pandas_df)
missing_value_df = pd.DataFrame({
    'column_name': pandas_df.columns, 'percent_missing': percent_missing
})
columns_to_sum_and_drop = missing_value_df.loc[missing_value_df['percent_missing'] >= 70]['column_name'].values.tolist()
pandas_df['summed_features'] = pandas_df[columns_to_sum_and_drop].sum(axis=1)
pandas_df.drop(labels=columns_to_sum_and_drop, axis=1, inplace=True)
# export to csv
pandas_df.to_csv('item_vectors_summed.csv', index=False)
