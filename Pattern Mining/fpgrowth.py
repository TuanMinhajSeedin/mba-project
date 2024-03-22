from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import collect_set
from pyspark.sql.functions import concat_ws


spark = SparkSession.builder \
                    .appName("AssociationRuleMining") \
                    .getOrCreate()

data = spark.read \
            .format("csv") \
            .option("header", "true") \
            .load("file:////Users/sacithrangana/Desktop/quiz/Data Mining Assignment/DMProject/mba-project/Pattern Mining/data/manipulated.csv", header=True, inferSchema=True)


transactions_df = data.groupBy("BillNo").agg(collect_set("ItemName").alias("items"))

fp_growth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.01)


model = fp_growth.fit(transactions_df)

frequent_itemsets = model.freqItemsets
association_rules = model.associationRules

frequent_itemsets.show()
association_rules.show()

spark.stop()


# transactions_df.write.csv("output.csv", header=True)

# # Concatenate array elements into a single string with a delimiter
# df_with_string_column = transactions_df.withColumn("items_string", concat_ws(",", "items"))

# # Write DataFrame with string column to CSV
# df_with_string_column.select("BillNo", "items_string").write.csv("output.csv", header=True)