from src.utils import *
from src.data_processing import load_data
from src.eda import basic_eda
from src.preprocessing import preprocess_data
from src.data_split import split_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
import logging

def main():
    # Load dataset
    df = load_data("final.csv")
    if df is None:
        logging.error("Dataset could not be loaded. Exiting.")
        return

    # Perform basic EDA
    basic_eda(df)

    # Preprocess the data
    df = preprocess_data(df)

    # Split the data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = split_data(df, "price")  # Ensure column name matches your dataset
    except Exception as e:
        logging.error("Data split failed: %s", e)
        return

    # Train the model
    try:
        model = train_model(X_train, y_train)
    except Exception as e:
        logging.error("Model training failed: %s", e)
        return

    # Evaluate the model
    try:
        mae, mse = evaluate_model(model, X_test, y_test)
        print(f"✅ Model Evaluation:\nMAE: {mae}\nMSE: {mse}")
    except Exception as e:
        logging.error("Model evaluation failed: %s", e)

if __name__ == "__main__":
    main()
