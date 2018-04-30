import sklearn.preprocessing
from fhub_transformers.missing import LagImputer
from fhub_transformers.ts import SingleLagger
from fhub_transformers import SimpleFunctionTransformer

groupby_kwargs = {'level': ['city', 'weekofyear']}

input = 'ndvi_se'

def my_function(data):
    return np.zeros((len(data), 3))

transformer = [
    SimpleFunctionTransformer(my_function)
]
feature = Feature(input=input, transformer=transformer)
