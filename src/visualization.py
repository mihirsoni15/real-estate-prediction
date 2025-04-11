import matplotlib.pyplot as plt
import logging

def plot_feature_distribution(df, feature):
    """
    Plots a histogram for a given feature in the DataFrame.
    """
    try:
        plt.figure()
        plt.hist(df[feature], bins=30, edgecolor='black')
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature}")
        plt.show()
    except Exception as e:
        logging.error("Error plotting feature distribution for %s: %s", feature, e)

def plot_scatter(df, x_feature, y_feature):
    """
    Plots a scatter plot between two features.
    """
    try:
        plt.figure()
        plt.scatter(df[x_feature], df[y_feature], alpha=0.5)
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f"{x_feature} vs {y_feature}")
        plt.show()
    except Exception as e:
        logging.error("Error plotting scatter plot for %s vs %s: %s", x_feature, y_feature, e)

def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap of the DataFrame features using matplotlib.
    """
    try:
        corr = df.corr()
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr, cmap='viridis', interpolation='none')
        plt.colorbar(im)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        plt.show()
    except Exception as e:
        logging.error("Error plotting correlation heatmap: %s", e)
