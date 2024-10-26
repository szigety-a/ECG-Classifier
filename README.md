ECG Classifier using Logistic Regression
Overview
This project implements a classifier for Electrocardiogram (ECG) data using Logistic Regression with Apache Spark's MLlib. The goal is to classify ECG signals into different categories based on labeled data. This tool is designed to handle large datasets efficiently using distributed computing.

Features
Data Preprocessing: Automatically reads and processes CSV data.
Label Indexing: Converts string labels to numeric indices for model training.
Feature Vectorization: Assembles feature columns into a single feature vector.
Model Training: Trains a Logistic Regression model with customizable hyperparameters.
Model Saving: Optionally saves the trained model for future use.
Evaluation Metrics: Calculates precision, recall, F1 score, overall accuracy, and displays the confusion matrix.
Metrics Saving: Optionally saves evaluation metrics to a text file.
Command-Line Interface: Easy to use with command-line arguments for flexibility.
Prerequisites
Python 3.6+
Apache Spark 2.4+
PySpark
Installation
Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/ecg-classifier.git
cd ecg-classifier
Install Dependencies

Make sure you have PySpark installed. If not, install it using:

bash
Copy code
pip install pyspark
Usage
Command-Line Arguments
input_file (required): Path to the input CSV file containing ECG data.
--maxIter (optional): Maximum number of iterations for Logistic Regression (default: 200).
--regParam (optional): Regularization parameter for Logistic Regression (default: 0.02).
--model_output (optional): Directory to save the trained model.
--metrics_output (optional): File path to save the evaluation metrics.
Running the Classifier
Prepare Your Data

Ensure your input CSV file is properly formatted. The first column should contain labels, and the subsequent columns should contain feature values.

Run the Script

Use spark-submit to run the script:

bash
Copy code
spark-submit ecg_classifier.py path/to/input.csv --maxIter 100 --regParam 0.01 --model_output path/to/save/model --metrics_output path/to/save/metrics.txt
Example:

bash
Copy code
spark-submit ecg_classifier.py data/ecg_data.csv --maxIter 150 --regParam 0.05 --model_output model/ecg_model --metrics_output results/metrics.txt
View Results

The script will output the following to the console:
Overall accuracy
Label index mapping
Precision, recall, and F1 score for each label
Confusion matrix
If --model_output is specified, the trained model will be saved to the provided directory.
If --metrics_output is specified, the evaluation metrics will be saved to the provided file.
Example Output
less
Copy code
Overall Accuracy: 0.9567

Label Index Mapping:
Label index 0: Label 'Normal'
Label index 1: Label 'Arrhythmia'
Label index 2: Label 'Other'

Metrics for Label Index 0 ('Normal'):
  Precision: 0.9700
  Recall:    0.9800
  F1 Score:  0.9750

Metrics for Label Index 1 ('Arrhythmia'):
  Precision: 0.9400
  Recall:    0.9300
  F1 Score:  0.9350

Metrics for Label Index 2 ('Other'):
  Precision: 0.9500
  Recall:    0.9400
  F1 Score:  0.9450

Confusion Matrix:
[[980.  10.  10.]
 [ 14. 930.  56.]
 [ 20.  15. 965.]]
Project Structure
bash
Copy code
ecg-classifier/
├── ecg_classifier.py      # Main script
├── data/
│   └── ecg_data.csv       # Sample data file (not included)
├── model/
│   └── ecg_model          # Directory for saved model
├── results/
    └── metrics.txt        # File for saved metrics
Customization
Hyperparameters: You can adjust the maxIter and regParam values to improve model performance.
Data Format: Modify the data reading section if your data format differs.
Additional Features: Feel free to extend the code to include additional evaluation metrics or model types.
Troubleshooting
Spark Configuration: Ensure that your Spark environment is properly configured.
Data Issues: Verify that your input data is clean and correctly formatted.
Dependencies: Make sure all dependencies are installed and compatible with your Python version.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Apache Spark: https://spark.apache.org/
PySpark Documentation: https://spark.apache.org/docs/latest/api/python/
Written by Andrew Szigety
