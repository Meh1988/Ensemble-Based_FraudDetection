# Ensemble-Based_FraudDetection
Ensemble-Based Deep Learning Classification for Fraud Detection
This repository contains a Python script that demonstrates an Ensemble-based deep learning classification approach for fraud detection using a CSV file with PCA-transformed features. The goal is to create an ensemble of neural networks to improve the accuracy and robustness of fraud detection.

Background
Fraud detection is a critical problem in various domains, including finance, online transactions, and insurance. Traditionally, rule-based systems and individual machine learning models are used. It is, however, possible to combine multiple models together and leverage their collective decision-making power by using Ensemble-based techniques.

Dataset
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only feature which has not been transformed with PCA is 'V29'. The feature 'V29' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

Due to the limitations on uploading large files to GitHub, we have made a decision to consider only 50,000 rows from the original dataset as the input file for our project. By selecting a subset of the data, we aim to comply with GitHub's file size restrictions while still maintaining a representative sample that allows us to perform exploratory data analysis, model training, and initial evaluations. Although this approach involves a reduction in the dataset size, we believe it will enable us to develop and share our code and results more effectively, facilitating collaboration and making it easier for others to reproduce our experiments. Additionally, we will provide clear instructions in the README on how to obtain the full dataset from an external source or request access to it for further investigations.
You can access the original file as below:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download



Methodology
Data Preprocessing:
The dataset is loaded, and the features and labels are separated.
The data is split into training and testing sets.
Standardization is applied to scale the features.

Ensemble Model:
A base neural network model is defined with multiple layers, including batch normalization and dropout for regularization.
Multiple instances of the base model are trained independently on the training data to create an ensemble of models.
Ensemble Prediction:

The ensemble of models makes predictions on the test set.
The individual predictions are combined using voting (majority vote) to obtain the final ensemble prediction.
Evaluation:

The accuracy and confusion matrix are computed to assess the performance of the Ensemble-based deep learning classification model.
How to Use
Ensure you have Python and the required libraries (NumPy, pandas, scikit-learn, TensorFlow, and Keras) installed.
Replace 'path_to_csv_file.csv' with the actual path to your CSV file containing the PCA-transformed features.
Run the Python script, and the ensemble model will be trained and evaluated on the provided dataset.
Results
The Ensemble-based deep learning classification model leverages the collective decision-making of multiple neural networks, leading to improved fraud detection accuracy and robustness. The accuracy and confusion matrix of the ensemble model on the test data are displayed.
