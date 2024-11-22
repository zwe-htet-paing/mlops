import pandas as pd
import numpy as np

def clean(
    df: pd.DataFrame,
    include_extreme_prices: bool = False,
) -> pd.DataFrame:

    # Handle missing values
    df = df.dropna(subset=['price'])
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    # Apply a log10 transformation to the target column
    df['price'] = np.log10(df['price'])

    # Calculate IQR
    Q1 = df['price'].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df['price'].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range

    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].reset_index(drop=True)

    return df