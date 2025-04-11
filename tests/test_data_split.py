import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from data_split import split_data


def test_split_data():
    df = pd.DataFrame({
        'price': [100, 200, 300, 400, 500],
        'feature': [1, 2, 3, 4, 5]
    })
    X_train, X_test, y_train, y_test = split_data(df, "price")
    
    # With test_size=0.2 and 5 records, expect 1 record for testing and 4 for training.
    assert len(X_test) == 1
    assert len(y_test) == 1
    assert len(X_train) == 4
