import pandas as pd
import numpy as np  
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.model_selection import train_test_split
import yaml

with open('Hyper-Params.yaml', 'r') as file:
    config = yaml.safe_load(file)

max_features = config['feature-engg']['max_features']

# Load and clean training data, dropping rows with missing 'content'
train_data = pd.read_csv('data/processed/train.csv').dropna(subset=['content'])
# Load and clean test data, dropping rows with missing 'content'
test_data = pd.read_csv('data/processed/test.csv').dropna(subset=['content'])

# Extract features (X) and labels (y) from training data
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

# Extract features (X) and labels (y) from test data
X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Initialize Bag of Words vectorizer
vectorizer = CountVectorizer(max_features=max_features)

# Fit vectorizer on training data and transform it to feature vectors
X_train_bow = vectorizer.fit_transform(X_train)

# Transform test data using the already fitted vectorizer
X_test_bow = vectorizer.transform(X_test)

# Convert training feature vectors to DataFrame
train_df = pd.DataFrame(X_train_bow.toarray())

# Add label column to training DataFrame
train_df['sentiment'] = y_train

# Convert test feature vectors to DataFrame
test_df = pd.DataFrame(X_test_bow.toarray())

# Add label column to test DataFrame
test_df['sentiment'] = y_test

# Save processed training features and labels to CSV
train_df.to_csv('data/interim/train_bow.csv', index=False)

# Save processed test features and labels
test_df.to_csv('data/interim/test_bow.csv', index=False)
