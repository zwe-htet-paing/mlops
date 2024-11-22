from typing import List, Optional

import pandas as pd
    
CATEGORICAL_FEATURES = ['room_type', 'neighbourhood']
NUMERICAL_FEATURES = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']


def select_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    if features:
        columns += features

    return df[columns]