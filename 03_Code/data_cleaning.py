import pandas as pd
import numpy as np
def clean_data(df):
    """
    Clean CICIoT2023 data.
    Steps:
    1. Replace positive and negative infinity with NaN.
    2. Remove rows containing any missing values.
    3. Keep duplicate rows unchanged.
    """
    original_rows = len(df)
    #Convert infinite values into missing values
    df = df.replace([np.inf, -np.inf], np.nan)
    #Remove rows with missing values
    df = df.dropna()
    #Count removed rows
    removed_rows = original_rows - len(df)
    return df, removed_rows