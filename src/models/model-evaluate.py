from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import pickle
import pandas as pd
import json

# Load the trained Random Forest model from disk
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the processed test data with Bag of Words features
test_data = pd.read_csv('data/interim/test_bow.csv')

# Extract features for testing (all columns except 'label')
x_test = test_data.drop(columns=['sentiment']).values

# Extract true labels for testing
y_test = test_data['sentiment'].values

# Predict labels using the trained model
y_pred = model.predict(x_test)

# Calculate evaluation metrics
metrics_dict = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred)
}

# Save the evaluation metrics to a JSON file for reporting
with open("reports/eval_metrics.json", "w") as metrics_file:
    json.dump(metrics_dict, metrics_file, indent=4)