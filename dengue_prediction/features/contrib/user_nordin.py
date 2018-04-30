import sklearn.preprocessing
from fhub_transformers.missing import LagImputer
from fhub_transformers.ts import SingleLagger

groupby_kwargs = {'level': ['city', 'weekofyear']}

input = 'ndvi_sw'
transformer = [
    LagImputer(groupby_kwargs=groupby_kwargs),
    SingleLagger(1, groupby_kwargs=groupby_kwargs),
    sklearn.preprocessing.Imputer(),
]
