from sklearn.linear_model import LinearRegression
import logging

def train_model(X_train, y_train):
    """
    Trains a Linear Regression model on the training data.
    Returns the trained model.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training the model: %s", e)
        raise e
