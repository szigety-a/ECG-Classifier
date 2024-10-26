from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics
import sys
import argparse
import os

#CS777 Final Project
#Andrew Szigety

def main():
    parser = argparse.ArgumentParser(description='ECG Classifier using Logistic Regression')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--maxIter', type=int, default=200, help='Maximum number of iterations for Logistic Regression')
    parser.add_argument('--regParam', type=float, default=0.02, help='Regularization parameter for Logistic Regression')
    parser.add_argument('--model_output', help='Directory to save the trained model')
    parser.add_argument('--metrics_output', help='File to save the evaluation metrics')

    args = parser.parse_args()

    spark = SparkSession.builder.appName("ECG-Classifier-LogisticRegression").getOrCreate()

    try:
        #Read the CSV file into a DataFrame
        df = spark.read.csv(args.input_file, header=False, inferSchema=True)

        #Rename columns: first column as 'label', rest as 'feature_0', 'feature_1', etc.
        old_columns = df.columns
        new_columns = ["label"] + ["feature_" + str(i) for i in range(len(old_columns)-1)]
        df = df.toDF(*new_columns)

        #Convert string labels to numeric indices
        indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
        indexer_model = indexer.fit(df)
        indexed_df = indexer_model.transform(df)
        indexed_labels = indexer_model.labels

        #Assemble feature columns into a feature vector
        assembler = VectorAssembler(inputCols=new_columns[1:], outputCol="features")
        assembled_df = assembler.transform(indexed_df)

        #Split the data into training and testing sets (80% training, 20% testing)
        train_df, test_df = assembled_df.randomSplit([0.8, 0.2], seed=42)

        #Initialize Logistic Regression model with hyperparameters
        lr = LogisticRegression(
            labelCol="labelIndex",
            featuresCol="features",
            maxIter=args.maxIter,
            regParam=args.regParam
        )

        #Train the model
        lr_model = lr.fit(train_df)

        #Save the trained model if an output directory is provided
        if args.model_output:
            lr_model.save(args.model_output)
            print(f"Model saved to {args.model_output}")

        #Make predictions on the test set
        predictions = lr_model.transform(test_df)

        #Select prediction and label columns
        prediction_and_labels = predictions.select("prediction", "labelIndex").rdd

        #Initialize MulticlassMetrics object
        metrics = MulticlassMetrics(prediction_and_labels)

        #Calculate overall accuracy
        accuracy = metrics.accuracy
        print(f"Overall Accuracy: {accuracy:.4f}\n")

        #Get unique labels
        labels = prediction_and_labels.map(lambda x: x[1]).distinct().collect()

        #Initialize dictionaries to hold metrics
        precision = {}
        recall = {}
        f1_score = {}

        #Calculate metrics for each label
        for label in sorted(labels):
            precision[label] = metrics.precision(label)
            recall[label] = metrics.recall(label)
            f1_score[label] = metrics.fMeasure(label)

        #Map label indices back to original labels
        label_mapping = {float(index): label for index, label in enumerate(indexed_labels)}

        #Print the label-index mapping
        print("Label Index Mapping:")
        for index, label in label_mapping.items():
            print(f"Label index {int(index)}: Label '{label}'")
        print()

        #Print metrics for each label
        for label in sorted(labels):
            print(f"Metrics for Label Index {int(label)} ('{label_mapping[label]}'):")
            print(f"  Precision: {precision[label]:.4f}")
            print(f"  Recall:    {recall[label]:.4f}")
            print(f"  F1 Score:  {f1_score[label]:.4f}\n")

        #Display the confusion matrix
        print("Confusion Matrix:")
        print(metrics.confusionMatrix().toArray())
        print()

        #Save metrics to a file if specified
        if args.metrics_output:
            with open(args.metrics_output, 'w') as f:
                f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
                f.write("Label Index Mapping:\n")
                for index, label in label_mapping.items():
                    f.write(f"Label index {int(index)}: Label '{label}'\n")
                f.write("\n")
                for label in sorted(labels):
                    f.write(f"Metrics for Label Index {int(label)} ('{label_mapping[label]}'):\n")
                    f.write(f"  Precision: {precision[label]:.4f}\n")
                    f.write(f"  Recall:    {recall[label]:.4f}\n")
                    f.write(f"  F1 Score:  {f1_score[label]:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(str(metrics.confusionMatrix().toArray()))
            print(f"Metrics saved to {args.metrics_output}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
