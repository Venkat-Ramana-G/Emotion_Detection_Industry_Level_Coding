import os
import logging
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple

# Configure logging
logging.basicConfig(
    filename='logs/features_engg.log',
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

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean train and test data, dropping rows with missing 'content'."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info(f"Loaded train data from {train_path} with shape {train_data.shape}")
        logging.info(f"Loaded test data from {test_path} with shape {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def extract_features_labels(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Extract features (X) and labels (y) from train and test data."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        logging.info("Extracted features and labels from train and test data.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error extracting features and labels: {e}")
        raise

def vectorize_text(
    X_train, X_test, max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame, CountVectorizer]:
    """Vectorize text data using Bag of Words."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        train_df = pd.DataFrame(X_train_bow.toarray())
        test_df = pd.DataFrame(X_test_bow.toarray())
        logging.info("Text data vectorized using Bag of Words.")
        return train_df, test_df, vectorizer
    except Exception as e:
        logging.error(f"Error during vectorization: {e}")
        raise

def save_features(
    train_df: pd.DataFrame, y_train, test_df: pd.DataFrame, y_test, output_dir: str
) -> None:
    """Save processed features and labels to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df['sentiment'] = y_train
        test_df['sentiment'] = y_test
        train_path = os.path.join(output_dir, 'train_bow.csv')
        test_path = os.path.join(output_dir, 'test_bow.csv')
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logging.info(f"Saved train features to {train_path} and test features to {test_path}")
    except Exception as e:
        logging.error(f"Error saving features: {e}")
        raise

def main() -> None:
    """Main function to run feature engineering pipeline."""
    try:
        config = load_config('Hyper-Params.yaml')
        max_features = config['feature-engg']['max_features']
        train_data, test_data = load_data('data/processed/train.csv', 'data/processed/test.csv')
        X_train, y_train, X_test, y_test = extract_features_labels(train_data, test_data)
        train_df, test_df, _ = vectorize_text(X_train, X_test, max_features)
        save_features(train_df, y_train, test_df, y_test, 'data/interim')
        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Feature engineering pipeline failed: {e}")

if __name__ == "__main__":
    main()  