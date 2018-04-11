import category_encoders
import pandas as pd
from fhub_transformers import NamedFramer, SimpleFunctionTransformer

# one-hot encoding of year

input = 'week_start_date'
transformer = [
    SimpleFunctionTransformer(
        lambda ser: pd.to_datetime(ser).dt.year
    ),
    NamedFramer(name='week_start_date'),
    category_encoders.OneHotEncoder(cols=['week_start_date']),
]
