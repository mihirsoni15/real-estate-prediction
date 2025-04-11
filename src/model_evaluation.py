from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
    Returns the MAE and MSE.
    """
    try:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        logging.info("Model evaluation completed.")
        return mae, mse
    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        raise e
