import pandas as pd
import logging

def preprocess_data(df):
    """
    Handles missing values and encodes categorical variables.
    Fills missing values with the mean and applies one-hot encoding.
    """
    try:
        df.fillna(df.mean(), inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        logging.info("Data preprocessing completed.")
        return df
    except Exception as e:
        logging.error("Error during preprocessing: %s", e)
        return df  # Optionally, you may choose to return None here.
