import os
import logging
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple

# Configure logging
logging.basicConfig(
    filename='logs/modelling.log',
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

def load_train_data(train_path: str) -> pd.DataFrame:
    """Load processed training data."""
    try:
        df = pd.read_csv(train_path)
        logging.info(f"Loaded training data from {train_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise

def get_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and labels from DataFrame."""
    try:
        X = df.drop(columns=['sentiment']).values
        y = df['sentiment'].values
        logging.info("Separated features and labels from training data.")
        return X, y
    except Exception as e:
        logging.error(f"Error separating features and labels: {e}")
        raise

def train_model(
    X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X, y)
        logging.info("Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Save the trained model to disk."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main() -> None:
    """Main function to run the modelling pipeline."""
    try:
        config = load_config('Hyper-Params.yaml')
        n_estimators = config['modelling']['n_estimators']
        max_depth = config['modelling']['max_depth']
        train_data = load_train_data('data/interim/train_bow.csv')
        X_train, y_train = get_features_labels(train_data)
        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, 'models/random_forest_model.pkl')
        logging.info("Modelling pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Modelling pipeline failed: {e}")

if __name__ == "__main__":
    main()