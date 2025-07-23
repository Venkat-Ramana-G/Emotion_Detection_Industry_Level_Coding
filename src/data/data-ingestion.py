# Import numpy library for numerical operations (not used in this script but commonly imported)
import numpy as np

# Import pandas library for data manipulation and analysis
import pandas as pd

# Import os library for interacting with the operating system (used for directory creation)
import os
import yaml

with open('Params.yaml', 'r') as file:
    config = yaml.safe_load(file)

test_size = config['data-ingestion']['test_size']   
# Import train_test_split function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Read the dataset from a remote CSV file into a pandas DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# Drop the 'tweet_id' column as it is not needed for analysis
df.drop(columns=['tweet_id'], inplace=True)

# Filter the DataFrame to include only rows where sentiment is 'happiness' or 'sadness'
final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]

# Replace sentiment labels: 'happiness' with 1 and 'sadness' with 0
final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

# Split the filtered DataFrame into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

# Create the directory 'data/raw' if it does not already exist
os.makedirs('data/raw', exist_ok=True)

# Save the training data to a CSV file in the 'data/raw' directory
train_data.to_csv('data/raw/train.csv', index=False)

# Save the testing data to a CSV file in the 'data/raw' directory
test_data.to_csv('data/raw/test.csv', index = False)