from fhub_transformers.base import SimpleFunctionTransformer
from fhub_transformers.missing import LagImputer
from sklearn.preprocessing import Imputer, StandardScaler

input = ['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw']
transformer = [
    LagImputer(groupby_kwargs={'level': 'city'}),
    Imputer(),
    StandardScaler(),
    SimpleFunctionTransformer(
        lambda df: np.mean(df, axis=1)
    )
]
