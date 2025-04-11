from sklearn.model_selection import train_test_split
import logging

def split_data(df, target_column):
    """
    Splits the dataset into training and testing sets.
    Returns X_train, X_test, y_train, y_test.
    """
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        logging.info("Data split into train and test sets.")
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        logging.error("Error splitting data: %s", e)
        raise e
