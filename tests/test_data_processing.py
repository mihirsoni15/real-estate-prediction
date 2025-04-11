import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd
from src.data_processing import load_data

def test_load_data_success(tmp_path):
    # Create a temporary CSV file
    data = {
        'price': [100, 200],
        'year_sold': [2000, 2001]
    }
    df_expected = pd.DataFrame(data)
    file_path = tmp_path / "temp.csv"
    df_expected.to_csv(file_path, index=False)

    # Load data using our function
    df_loaded = load_data(str(file_path))
    pd.testing.assert_frame_equal(df_expected, df_loaded)

def test_load_data_failure():
    # Try to load a non-existent file
    df_loaded = load_data("non_existing_file.csv")
    assert df_loaded is None
