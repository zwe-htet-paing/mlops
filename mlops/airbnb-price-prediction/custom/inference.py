import numpy as np
from typing import Dict, List, Tuple, Union

from sklearn.feature_extraction import DictVectorizer
from xgboost import Booster

from mlops.utils.models.xgboost import build_data

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

DEFAULT_INPUTS = [
    {   
        # target = "price"
        'room_type': 'Private room',
        'neighbourhood': 'FOURTEENTH WARD',
        'latitude': 42.66719,
        'longitude': -73.8158,
        'minimum_nights': 1,
        'number_of_reviews': 250,
        'reviews_per_month': 1.93,
        'availability_365': 255
    },
    {
        # target = "price"
        'room_type': 'Private room',
        'neighbourhood': 'FOURTEENTH WARD',
        'latitude': 42.66719,
        'longitude': -73.8158,
        'minimum_nights': 1,
        'number_of_reviews': 250,
        'reviews_per_month': 1.93,
        'availability_365': 255
    },
]


@custom
def predict(
    model_settings: Dict[str, Tuple[Booster, DictVectorizer]],
    **kwargs,
) -> List[float]:
    inputs:  List[Dict[str, Union[str, float, int]]] = kwargs.get('inputs', DEFAULT_INPUTS)

    # print(inputs)

    room_type = kwargs.get('room_type')
    neighbourhood = kwargs.get('neighbourhood')
    latitude = kwargs.get('latitude')
    longitude = kwargs.get('longitude')
    minimum_nights = kwargs.get('minmum_nights')
    number_of_reviews = kwargs.get('number_of_reviews')
    reviews_per_month = kwargs.get('reviews_per_month')
    availability_365 = kwargs.get('availability_365')


    if room_type is not None or neighbourhood is not None or latitude is not None \
        or longitude is not None or minimum_nights is not None or number_of_reviews is not None \
        or reviews_per_month is not None or availability_365 is not None:
        inputs = [
            {
                'room_type': room_type,
                'neighbourhood': neighbourhood,
                'latitude': latitude,
                'longitude': longitude,
                'minimum_nights': minimum_nights,
                'number_of_reviews': number_of_reviews,
                'reviews_per_month': reviews_per_month,
                'availability_365': availability_365
            },
        ]
    
    model, vectorizer = model_settings['xgboost']
    vectors = vectorizer.transform(inputs)

    predictions = model.predict(build_data(vectors))
    predictions = np.power(10, predictions)

    for idx, input_feature in enumerate(inputs):
        print(f'Prediction of price using these features: {predictions[idx]}')
        for key, value in inputs[idx].items():
            print(f'\t{key}: {value}')

    return predictions.tolist()