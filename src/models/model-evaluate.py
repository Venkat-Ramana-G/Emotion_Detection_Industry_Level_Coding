import os
import logging
import pickle
import pandas as pd
import json
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    filename='logs/model_evaluate.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_model(model_path: str) -> Any:
    """Load a trained model from disk."""
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_test_data(test_path: str) -> pd.DataFrame:
    """Load processed test data."""
    try:
        df = pd.read_csv(test_path)
        logging.info(f"Loaded test data from {test_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

def get_features_labels(df: pd.DataFrame) -> tuple:
    """Extract features and labels from test DataFrame."""
    try:
        X = df.drop(columns=['sentiment']).values
        y = df['sentiment'].values
        logging.info("Extracted features and labels from test data.")
        return X, y
    except Exception as e:
        logging.error(f"Error extracting features and labels: {e}")
        raise

def evaluate_model(model: Any, X_test, y_test) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info(f"Evaluation metrics calculated: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        logging.info(f"Saved evaluation metrics to {output_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def main() -> None:
    """Main function to run model evaluation pipeline."""
    try:
        model = load_model('models/random_forest_model.pkl')
        test_data = load_test_data('data/interim/test_bow.csv')
        X_test, y_test = get_features_labels(test_data)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/eval_metrics.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()