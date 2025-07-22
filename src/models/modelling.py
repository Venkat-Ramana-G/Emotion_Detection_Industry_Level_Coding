import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier

# Load the processed training data with Bag of Words features
train_data = pd.read_csv('data/interim/train_bow.csv')

# Separate features and sentiment for training
x_train = train_data.drop(columns=['sentiment']).values
y_train = train_data['sentiment'].values

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model on the training data
model.fit(x_train, y_train)

# Save the trained model to disk using pickle
with open('models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)