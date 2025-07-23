import os
import yaml
import logging

with open('Hyper-Params.yaml', 'r') as file:
    config = yaml.safe_load(file)

test_size = config['data-ingestion']['test_size']   
# Import train_test_split function to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    filename='logs/data_ingestion.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise

def fetch_data(url: str) -> pd.DataFrame:
    """Fetch dataset from a remote CSV file."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Data fetched from {url} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error fetching data from {url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame: drop columns, filter, and encode labels."""
    try:
        df = df.drop(columns=['tweet_id'])
        logging.info("Dropped 'tweet_id' column.")

        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        logging.info("Filtered for 'happiness' and 'sadness' sentiments.")
        
        df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Encoded sentiments: 'happiness'->1, 'sadness'->0.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train and test sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Split data into train ({train_data.shape}) and test ({test_data.shape})")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """Save train and test DataFrames to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, 'train.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Saved train data to {train_path} and test data to {test_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    try:
        config = load_config('Hyper-Params.yaml')
        test_size = config['data-ingestion']['test_size']
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = fetch_data(url)
        final_df = preprocess_data(df)
        train_data, test_data = split_data(final_df, test_size)
        save_data(train_data, test_data, 'data/raw')
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")

if __name__ == '__main__': 
    main()