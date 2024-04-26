# %% [markdown]
#  ## Machine Learning

# %%
# !pip install pyspark

# %% [markdown]
# ## Imports

# %%

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import col, countDistinct, isnan, when, count, round, substring_index,substring, split, regexp_replace, udf
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator,ClusteringEvaluator
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LinearSVC,GBTClassifier
from pyspark.ml.clustering import KMeans

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tabulate import tabulate

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %% [markdown]
# ## Functions

# %%
def evaluate_model(model, train_data, validation_data, test_data, model_name, label_column):
    # Evaluate training data
    train_predictions = model.transform(train_data)
    train_metrics = calculate_metrics(train_predictions, label_column)

    # Evaluate validation data
    validation_predictions = model.transform(validation_data)
    validation_metrics = calculate_metrics(validation_predictions, label_column)

    # Evaluate test data
    test_predictions = model.transform(test_data)
    test_metrics = calculate_metrics(test_predictions, label_column)

     # Prepare data for tabulate
    table = [
        ['Model', model_name,'',''],
        ['Metric', 'Training', 'Validation', 'Test'],
        ['Accuracy', train_metrics['accuracy'], validation_metrics['accuracy'], test_metrics['accuracy']],
        ['Weighted Precision', train_metrics['weighted_precision'], validation_metrics['weighted_precision'], test_metrics['weighted_precision']],
        ['Weighted Recall', train_metrics['weighted_recall'], validation_metrics['weighted_recall'], test_metrics['weighted_recall']],
        ['F1 Score', train_metrics['f1'], validation_metrics['f1'], test_metrics['f1']]
    ]

    # Display results using tabulate
    print(tabulate(table, headers="firstrow", tablefmt='grid'))

def calculate_metrics(predictions, label_column):
    evaluator_multi = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol=label_column, metricName='accuracy')
    evaluator_weighted_precision = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol=label_column, metricName='weightedPrecision')
    evaluator_weighted_recall = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol=label_column, metricName='weightedRecall')
    evaluator_f1 = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol=label_column, metricName='f1')

    accuracy = evaluator_multi.evaluate(predictions)
    weighted_precision = evaluator_weighted_precision.evaluate(predictions)
    weighted_recall = evaluator_weighted_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    return {
        'accuracy': accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'f1': f1
    }

# %%

spark=SparkSession.builder\
    .master("local[*]")\
    .appName("LoanApproval")\
    .getOrCreate()


# %%

sc=spark.sparkContext


# %% [markdown]
#  ## Read Data - SBAnational.csv

# %%

# data_path="/content/drive/MyDrive/Colab Notebooks/BD_project/50000.csv"
data_path="../../data/SBAnational.csv"


# %%

loan_df =  spark.read.csv(data_path, header=True, inferSchema=True, multiLine=True, quote='"', escape='"')


# %%
loan_df.printSchema()
loan_df.show(5)

# %%

print("Transforming categorial features...")
# List of categorical columns to be one-hot encoded
categorical_columns = ["Name", "City", "State", "Zip", "Bank", "BankState", "UrbanRural", "RevLineCr", "LowDoc", "Sector", "ApprovalMonth"]
# ======================================================
# dropping these columns give better accuracy (by trial)
# ======================================================
loan_df = loan_df.drop('Name')
categorical_columns = ["City", "State", "Zip", "Bank", "BankState", "UrbanRural", "RevLineCr", "LowDoc", "Sector", "ApprovalMonth"]

loan_df = loan_df.drop('Zip')
categorical_columns = ["City", "State", "Bank", "BankState", "UrbanRural", "RevLineCr", "LowDoc", "Sector", "ApprovalMonth"]

loan_df = loan_df.drop('City')
categorical_columns = ["State", "Bank", "BankState", "UrbanRural", "RevLineCr", "LowDoc", "Sector", "ApprovalMonth"]
# ======================================================
# ======================================================
# ======================================================

# Define an empty list to store the pipeline stages
stages = []

# Iterate over each categorical column
for column in categorical_columns:
    # Define StringIndexer for the current column
    indexer = StringIndexer(inputCol=column, outputCol=column + "Index")

    # Define OneHotEncoder for the indexed column
    encoder = OneHotEncoder(inputCol=column + "Index", outputCol=column + "Vec")

    # Add StringIndexer and OneHotEncoder to the list of stages
    stages += [indexer, encoder]
label_column = "MIS_Status"



# Create VectorAssembler for combining all features
# List of input columns (excluding the label column and categorical columns)
input_columns = [col for col in loan_df.columns if col != label_column and col not in categorical_columns]
input_columns += [column + "Vec" for column in categorical_columns]
assembler = VectorAssembler(inputCols=input_columns , outputCol="features")

# Combine all stages into a Pipeline
pipeline = Pipeline(stages=stages + [assembler])

# Fit the pipeline to your data
pipeline_model = pipeline.fit(loan_df)

# Transform your data using the pipeline
transformed_data = pipeline_model.transform(loan_df)
transformed_data.show(5)
print("Splitting data into training, validation and test...")
# Split the transformed data into training and test sets (70% training, 30% test)
# (trainingData, testData) = transformed_data.randomSplit([0.7, 0.3])
(trainingData, validationData, testData) = transformed_data.randomSplit([0.6, 0.2, 0.2], seed=123)

# %% [markdown]
# ## Logistic Regression

# %%
# Create a Logistic Regression model
lr = LogisticRegression(maxIter=10, elasticNetParam=0.8, labelCol=label_column, featuresCol="features")
print("Training logistic regression model...")
# Train the model
lrModel = lr.fit(trainingData)

# %%

# Make predictions on the test data
predictions = lrModel.transform(validationData)

# predictions.describe().show()
# Evaluate the model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol=label_column)
accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)


# %%
evaluate_model(lrModel, trainingData, validationData, testData, 'Logistic Regression', label_column)

# %% [markdown]
# ## Random Forest

# %%

 # Create Random Forest model
rf = RandomForestClassifier(featuresCol='features', labelCol=label_column)

# Fit model to training data
rf_model = rf.fit(trainingData)




# %%

evaluate_model(rf_model, trainingData, validationData, testData, 'Random Forest', label_column)


# %% [markdown]
# ## GBTClassifier

# %%

# Split the data into training and test sets (30% held out for testing)
# (trainingData, testData) = transformed_data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTClassifier(featuresCol='features', labelCol=label_column, maxIter=100)
print("Training...")
# Train model.  This also runs the indexers.
gbt_model = gbt.fit(trainingData)

print("Evaluating...")
evaluate_model(gbt_model, trainingData, validationData, testData, 'GBTClassifier', label_column)


# %% [markdown]
# ## SVM

# %%
lsvc = LinearSVC(featuresCol='features', labelCol=label_column,maxIter=100, regParam=0.1)
print("Training...")
# Fit the model
lsvcModel = lsvc.fit(trainingData)
print("Evaluating...")
evaluate_model(lsvcModel, trainingData, validationData, testData, 'SVM', label_column)


# %% [markdown]
#  ## Save

# %%
# model_path = "lrModel"
# lrModel.save(model_path)




