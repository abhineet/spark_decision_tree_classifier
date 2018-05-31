
"""Basic Example of using 
decision tree classifier using pyspark

Usage : spark-submit decision_tree_cl_pyspark.py data/iris.data

Author: 
Abhineet Verma

"""

## Imports

from pyspark import SparkConf, SparkContext

from operator import add
import sys
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


## Constants
APP_NAME = "Decision Tree Classification Example"
##OTHER FUNCTIONS/CLASSES

def main(spark,filename):
  df = spark.read.csv(filename,header=False,inferSchema=True)
  vector_assembler = VectorAssembler(inputCols=['_c0','_c1','_c2','_c3'],outputCol='features')
# df.show(4)
# +---+---+---+---+-----------+
# |_c0|_c1|_c2|_c3|        _c4|
# +---+---+---+---+-----------+
# |5.1|3.5|1.4|0.2|Iris-setosa|
# |4.9|3.0|1.4|0.2|Iris-setosa|
# |4.7|3.2|1.3|0.2|Iris-setosa|
# |4.6|3.1|1.5|0.2|Iris-setosa|
# +---+---+---+---+-----------+
  vector_assembler = VectorAssembler(inputCols=['_c0','_c1','_c2','_c3'],outputCol='features')
  v_df = vector_assembler.transform(df)

# v_df.show(4)
# +---+---+---+---+-----------+-----------------+
# |_c0|_c1|_c2|_c3|        _c4|         features|
# +---+---+---+---+-----------+-----------------+
# |5.1|3.5|1.4|0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|
# |4.9|3.0|1.4|0.2|Iris-setosa|[4.9,3.0,1.4,0.2]|
# |4.7|3.2|1.3|0.2|Iris-setosa|[4.7,3.2,1.3,0.2]|
# |4.6|3.1|1.5|0.2|Iris-setosa|[4.6,3.1,1.5,0.2]|
# +---+---+---+---+-----------+-----------------+
# only showing top 4 rows
  indexer = StringIndexer(inputCol='_c4',outputCol='label')
  i_df = indexer.fit(v_df).transform(v_df)
#   i_df.show(4)
# +---+---+---+---+-----------+-----------------+-----+
# |_c0|_c1|_c2|_c3|        _c4|         features|label|
# +---+---+---+---+-----------+-----------------+-----+
# |5.1|3.5|1.4|0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|  0.0|
# |4.9|3.0|1.4|0.2|Iris-setosa|[4.9,3.0,1.4,0.2]|  0.0|
# |4.7|3.2|1.3|0.2|Iris-setosa|[4.7,3.2,1.3,0.2]|  0.0|
# |4.6|3.1|1.5|0.2|Iris-setosa|[4.6,3.1,1.5,0.2]|  0.0|
# +---+---+---+---+-----------+-----------------+-----+
# only showing top 4 rows
  splits = i_df.randomSplit([0.6,0.4],1)
  train_df =  splits[0]
  test_df = splits[1]
  dt = DecisionTreeClassifier(labelCol= 'label',featuresCol='features')
  dt_model = dt.fit(train_df)
  dt_pred = dt_model.transform(test_df)
  dt_evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
  dt_accuracy =  dt_evaluator.evaluate(dt_pred)
  print(dt_accuracy)
  #0.931034482759
  
 

if __name__ == "__main__":

   # Configure Spark
   # conf = SparkConf().setAppName(APP_NAME)
   # conf = conf.setMaster("local[*]")
   # sc   = SparkContext(conf=conf)
   filename = sys.argv[1]
   spark = SparkSession\
        .builder\
        .appName(APP_NAME)\
        .getOrCreate()
   # Execute Main functionality
   main(spark, filename)
