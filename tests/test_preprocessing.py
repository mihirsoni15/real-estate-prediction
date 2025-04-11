import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data

def test_preprocess_data():
    # Create a dummy dataframe with missing values and a categorical variable
    df = pd.DataFrame({
        'price': [100, np.nan, 300],
        'category': ['A', 'B', 'A']
    })
    df_processed = preprocess_data(df.copy())
    
    # Ensure no missing values remain in numeric columns
    assert df_processed['price'].isnull().sum() == 0
    
    # Check that one-hot encoding is applied (should have a column for category_B, as drop_first=True)
    assert 'category_B' in df_processed.columns
