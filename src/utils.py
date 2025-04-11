import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging (this is our central logging configuration)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure pandas and matplotlib settings
pd.set_option('display.max_columns', 50)
plt.style.use('fivethirtyeight')    