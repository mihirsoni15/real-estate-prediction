import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import pandas as pd

def test_train_model():
    # Create a small dataset
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [2, 3, 4, 5]
    })
    y_train = pd.Series([10, 20, 30, 40])
    
    model = train_model(X_train, y_train)
    
    # Check that the returned model has a predict method
    assert hasattr(model, 'predict')
