import category_encoders
import pandas as pd

from dengue_prediction.features.feature import Feature
from dengue_prediction.features.transformers import SimpleFunctionTransformer, NamedFramer

input='week_start_date'
transformer=[
    SimpleFunctionTransformer(
        lambda ser: pd.to_datetime(ser).dt.year
    ),
    NamedFramer(name='week_start_date'),
    category_encoders.OneHotEncoder(cols=['week_start_date']),
]
