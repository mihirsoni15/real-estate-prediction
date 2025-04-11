from src.data_processing import load_data
from src.visualization import plot_feature_distribution, plot_scatter, plot_correlation_heatmap
import logging

def main():
    # Load dataset
    df = load_data("final.csv")
    if df is None:
        logging.error("Dataset not loaded.")
        return
    
    # Plot distribution of the 'price' feature
    plot_feature_distribution(df, "price")
    
    # Plot scatter between 'sqft' and 'price'
    plot_scatter(df, "sqft", "price")
    
    # Plot the correlation heatmap of all features
    plot_correlation_heatmap(df)

if __name__ == "__main__":
    main()
