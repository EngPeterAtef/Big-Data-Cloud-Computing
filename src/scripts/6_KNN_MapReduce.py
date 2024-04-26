from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import col, countDistinct, isnan, struct
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator,ClusteringEvaluator
from tabulate import tabulate
import numpy as np

# %%
spark=SparkSession.builder\
    .master("local[*]")\
    .appName("LoanApproval")\
    .getOrCreate()

# %%
sc=spark.sparkContext

# %%
# data_path="../../data/preprocessed.csv"
data_path="../../sample_data/50000.csv"
loan_df =  spark.read.csv(data_path, header=True, inferSchema=True, multiLine=True, quote='"', escape='"')
loan_df = loan_df.drop('Name')
loan_df = loan_df.drop('Zip')
loan_df = loan_df.drop('City')
loan_df.printSchema()
loan_df.show(5)

# %%

print("Transforming categorial features...")
# List of categorical columns to be one-hot encoded


categorical_columns = ["State", "Bank", "BankState", "UrbanRural", "Sector", "ApprovalMonth"]
# ======================================================
# ======================================================
# ======================================================
# Define an empty list to store the pipeline stages
stages = []

# Fit StringIndexer on the entire dataset and add to stages
indexers = [StringIndexer(inputCol=column, outputCol=column + "Index").fit(loan_df) for column in categorical_columns]
stages += indexers

# Define OneHotEncoder for the indexed columns
encoders = [OneHotEncoder(inputCol=column + "Index", outputCol=column + "Vec", dropLast=False) for column in categorical_columns]
stages += encoders


label_column = "MIS_Status"

# Create VectorAssembler for combining all features
# List of input columns (excluding the label column and categorical columns)
input_columns = [col for col in loan_df.columns if col != label_column and col not in categorical_columns]
input_columns += [column + "Vec" for column in categorical_columns]
assembler = VectorAssembler(inputCols=input_columns, outputCol="features")

# Combine all stages into a Pipeline
pipeline = Pipeline(stages=stages + [assembler])

# Fit the pipeline to your data
pipeline_model = pipeline.fit(loan_df)

# Transform your data using the pipeline
transformed_data = pipeline_model.transform(loan_df)
transformed_data.show(5)

print("Splitting data into training and test...")
(trainingData, testData) = transformed_data.randomSplit([0.8, 0.2], seed=123)

# %% [markdown]
# ## Make a new DF that has only MIS_Status and features

# %%
training_df = trainingData.select("MIS_Status", "features")
test_df = testData.select("MIS_Status", "features")
training_df.show(5,truncate=False)

# %% [markdown]
# ## Convert to RDD to Apply MapReduce

# %%
training_rdd = training_df.rdd
test_rdd = test_df.rdd
print(training_rdd.take(5))

# %%
print("Number of partitions before repartitioning:", training_rdd.getNumPartitions())
# Repartition the RDD into a new number of partitions
num_partitions = 3  # Change this to the desired number of partitions
training_rdd = training_rdd.repartition(num_partitions)
# New number of partitions
print("Number of partitions after repartitioning:", training_rdd.getNumPartitions())


# %%
# Collect the elements of the RDD into a list
test_list = test_rdd.collect()

# %% [markdown]
# # KNN

# %%
def appy_knn(rdd, query_point, k):
  def cosine_similarity(np_vector1, np_vector2):
    # Compute dot product
    dot_product = np.dot(np_vector1, np_vector2)

    # Compute magnitudes
    mag1 = np.sqrt(np.sum(np_vector1 ** 2))
    mag2 = np.sqrt(np.sum(np_vector2 ** 2))

    # Handle division by zero
    if mag1 == 0 or mag2 == 0:
        return 0
    # Compute cosine similarity
    return dot_product / (mag1 * mag2)

  def map_phase(split):
      """Map phase: Find k-nearest neighbors in each split."""
      neighbors = []
      for row in split:
          true_class = row.MIS_Status
          data_point = row.features
          # Convert PySpark sparse vectors to NumPy arrays
          np_vector1 = np.array(query_point.toArray())
          np_vector2 = np.array(data_point.toArray())
          # Calculate cosine similarity
          dist = cosine_similarity(np_vector1, np_vector2)
          neighbors.append((None, {'similarity': dist, 'class': true_class}))
      # Sort the neighbors by similarity
      neighbors.sort(key=lambda x: x[1]['similarity'], reverse=True)
      # Take the top k neighbors
      k_neighbors = neighbors[:k]

      return [k_neighbors]

  def reduce_phase(neighbors1, neighbors2):
      """Reduce phase: Find the definitive top k neighbors."""
      # Merge the neighbors from different splits
      merged_neighbors = neighbors1 + neighbors2
      # Sort the merged neighbors by distance
      merged_neighbors.sort(key=lambda x: x[1]['similarity'], reverse=True)
      # Take the top k neighbors
      return merged_neighbors[:k]

  def classify_input(data):
    # Extract the classes from the data
    classes = np.array([entry[1]['class'] for entry in data])

    # Count the occurrences of each class
    class_counts = np.bincount(classes)

    # Find the most common class
    most_common_class = np.argmax(class_counts)

    # print("Most frequent class:", most_common_class)
    return most_common_class

  # Map phase: Apply map transformation to each split of the training data
  mapped_neighbors = rdd.mapPartitions(map_phase)
  # print("mapped_neighbors_rdd")
  # print("Number of partitions:", mapped_neighbors.getNumPartitions())
  # print(mapped_neighbors.take(10))

  # Reduce phase: Aggregate results from the map phase using reduce
  final_neighbors = mapped_neighbors.reduce(reduce_phase)
  # print("Final K Nearest Neighbors:", final_neighbors)
  return classify_input(final_neighbors)


# %% [markdown]
# # EVALUATION

# %% [markdown]
# 

# %%
def calculate_confusion_matrix(true_labels, predicted_labels, labels):
    num_classes = len(labels)
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    label_to_index = {label: i for i, label in enumerate(labels)}
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_index = label_to_index[true_label]
        predicted_index = label_to_index[predicted_label]
        confusion_matrix[true_index][predicted_index] += 1
    return confusion_matrix


def calculate_accuracy(confusion_matrix):
    tp_tn = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    tp_tn_fp_fn = sum(sum(row) for row in confusion_matrix)
    accuracy = tp_tn / tp_tn_fp_fn
    return accuracy

def calculate_f1_score(precision, recall):
    return 2* (precision * recall) / (precision + recall)

def calculate_precision(confusion_matrix, class_index):
    # Calculate precision for a specific class
    true_positive = confusion_matrix[class_index][class_index]
    column_sum = sum(confusion_matrix[i][class_index] for i in range(len(confusion_matrix)))
    precision = true_positive / column_sum if column_sum != 0 else 0
    return precision

def calculate_recall(confusion_matrix, class_index):
    # Calculate recall for a specific class
    true_positive = confusion_matrix[class_index][class_index]
    row_sum = sum(confusion_matrix[class_index])
    recall = true_positive / row_sum if row_sum != 0 else 0
    return recall

def calculate_macro_average_precision(confusion_matrix):
    num_classes = len(confusion_matrix)
    precisions = [calculate_precision(confusion_matrix, i) for i in range(num_classes)]
    macro_average_precision = sum(precisions) / num_classes
    return macro_average_precision

def calculate_macro_average_recall(confusion_matrix):
    num_classes = len(confusion_matrix)
    recalls = [calculate_recall(confusion_matrix, i) for i in range(num_classes)]
    macro_average_recall = sum(recalls) / num_classes
    return macro_average_recall


def calculate_micro_average_precision(confusion_matrix):
    num_classes = len(confusion_matrix)
    true_positives = sum(confusion_matrix[i][i] for i in range(num_classes))
    all_positives = sum(sum(confusion_matrix[i]) for i in range(num_classes))
    micro_average_precision = true_positives / all_positives if all_positives != 0 else 0
    return micro_average_precision

def calculate_micro_average_recall(confusion_matrix):
    num_classes = len(confusion_matrix)
    true_positives = sum(confusion_matrix[i][i] for i in range(num_classes))
    all_actuals = sum(sum(row) for row in confusion_matrix)
    micro_average_recall = true_positives / all_actuals if all_actuals != 0 else 0
    return micro_average_recall
def display_confusion_matrix(confusion_matrix, labels):
     # Prepare data for tabulate
    table = [
        ['', *labels],
        [labels[0],*confusion_matrix[0]],
        [labels[1],*confusion_matrix[1]]
    ]
    # Display results using tabulate
    print(tabulate(table, headers="firstrow", tablefmt='grid'))

def evaluate_knn(test_list):
  predicted_list = []
  true_list = []
  for row in test_list:
    true_label = row.MIS_Status
    point = row.features
    knn_predict = appy_knn(training_rdd, point, 3)
    predicted_list.append(knn_predict)
    true_list.append(true_label)
  # print(true_label)
  labels = [1,0]
  # Calculate accuracy manually
  confusion_matrix= calculate_confusion_matrix(true_list, predicted_list, labels=labels)
  print("Confusion Matrix:")
  display_confusion_matrix(confusion_matrix, labels)
  accuracy = calculate_accuracy(confusion_matrix)
  print("Accuracy:", accuracy)

  macro_precision = calculate_macro_average_precision(confusion_matrix)
  print("macro_precision:", macro_precision)
  micro_precision = calculate_micro_average_precision(confusion_matrix)
  print("micro_precision:", micro_precision)

  macro_recall = calculate_macro_average_recall(confusion_matrix)
  print("macro_recall:", macro_recall)
  micro_recall = calculate_micro_average_recall(confusion_matrix)
  print("micro_recall:", micro_recall)

  f1_macro = calculate_f1_score(macro_precision,macro_recall)
  print("f1_macro:", f1_macro)
  f1_micro = calculate_f1_score(micro_precision,micro_recall)
  print("f1_micro:", f1_micro)

# %%
evaluate_knn(test_list)

# %%



