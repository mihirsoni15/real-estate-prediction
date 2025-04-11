import pandas as pd
import logging

def load_data(file_path):
    """
    Loads dataset from a CSV file and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully from %s", file_path)
        return df
    except Exception as e:
        logging.error("Error loading data from %s: %s", file_path, e)
        return None
