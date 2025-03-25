# Amazon Review Classifier

## Overview
This repository contains the implementation of a machine learning model for classifying Amazon reviews into one of six categories:
- Product Quality
- Shipping
- Price
- Usability
- Customer Service
- Aesthetic Appeal

This project was developed as part of Group 16's final project for COMPSCI 4NL3 (Natural Language Processing, Data Collection, and Annotation).

## Project Structure
The repository includes the following files:
- `logistic_regression_model.pkl`: Model file for logistic regression.
- `main.py`: Python script containing the implementation of the classification models.
- `test_set.csv`: Test dataset for evaluation.
- `test_predictions.csv`: Predicted labels for the test dataset.
- `tfidf_vectorizer.pkl`: Vectorizer file for TF-IDF transformation.
- `train_set.csv`: Training dataset.
- `validation_set.csv`: Validation dataset.

## Models Implemented
1. **Logistic Regression**: A linear model for classification tasks.
2. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.

## Setup for Codabench
The repository is configured for compatibility with Codabench, allowing for easy evaluation and benchmarking of the models.

## Usage
1. **Training the Model**:
   - Use `train.csv` to train the model.
   - Example command:
     ```bash
     python model.py --train --data train.csv
     ```

2. **Validation**:
   - Use `validation.csv` to validate the model's performance.
   - Example command:
     ```bash
     python model.py --validate --data validation.csv
     ```

3. **Testing**:
   - Use `test.csv` to generate predictions.
   - Example command:
     ```bash
     python model.py --test --data test.csv
     ```

## Results
The predictions for the test dataset are stored in `test_predictions.csv`.

## Contributors
- [Richard Nguyen](https://github.com/richnguyenn)
- [Anthony Hana](https://github.com/anthonyhana04)
- [Alan Zhou](https://github.com/azowmann)
