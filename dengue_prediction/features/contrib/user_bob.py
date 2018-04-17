from fhub_transformers.missing import LagImputer
from fhub_transformers.ts import SingleLagger
from sklearn.preprocessing import Imputer

groupby_kwargs = {'level': ['city', 'weekofyear']}

input = 'reanalysis_max_air_temp_k'
transformer = [
    LagImputer(groupby_kwargs=groupby_kwargs),
    SingleLagger(2, groupby_kwargs=groupby_kwargs),
    Imputer(),
]
