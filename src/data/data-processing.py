import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Configure logging
logging.basicConfig(
    filename="logs/data_processing.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df: pd.DataFrame) -> None:
    """Set text to NaN if sentence has fewer than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the 'content' column of the DataFrame."""
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        logging.info("Text normalization completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during text normalization: {e}")
        raise

def normalized_sentence(sentence: str) -> str:
    """Apply all preprocessing steps to a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Error normalizing sentence: {e}")
        raise

def load_and_normalize_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test data, normalize, and return them."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info(f"Loaded train data from {train_path} with shape {train_data.shape}")
        logging.info(f"Loaded test data from {test_path} with shape {test_data.shape}")

        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading or normalizing data: {e}")
        raise

def save_processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """Save processed train and test data to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Processed train data saved to {train_path}")
        logging.info(f"Processed test data saved to {test_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise

def main() -> None:
    """Main function to run the data processing pipeline."""
    try:
        train_data, test_data = load_and_normalize_data("data/raw/train.csv", "data/raw/test.csv")
        save_processed_data(train_data, test_data, "data/processed")
        logging.info("Data processing pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Data processing pipeline failed: {e}")

if __name__ == "__main__":
    main()