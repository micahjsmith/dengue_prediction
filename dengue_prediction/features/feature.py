import logging


import funcy
import numpy as np
import pandas as pd
from sklearn.pipeline import _name_estimators
from sklearn_pandas.pipeline import TransformerPipeline


from dengue_prediction.util import asarray2d


logger = logging.getLogger(__name__)


__all__ = ['Feature', 'FeatureValidator']


class RobustTransformerPipeline(TransformerPipeline):
    
    def transform(self, X, *args, **kwargs):
        _transform = _make_robust_to_tabular_types(
            super().transform, 1)
        return _transform(X, *args, **kwargs)
    
    
def make_robust_transformer_pipeline(*steps):
    """Construct a TransformerPipeline from the given estimators.
    """
    return RobustTransformerPipeline(_name_estimators(steps))


def _make_approaches(funcs, catch):
    if funcs:
        for func in funcs[:-1]:
            yield func, catch 
        yield funcs[-1], ()
        

def _make_robust_to_tabular_types(func, n):
    @funcy.wraps(func)
    def wrapped(*args, **kwargs):
        approaches = _make_approaches(
            (funcy.identity,
             pd.DataFrame,
             np.asarray,
             asarray2d),
            (ValueError, TypeError) 
        )
        nonconverted_args = args[n:]
        for convert, catch in approaches:
            try:
                converted_args = [convert(args[j]) for j in range(n)]
                args_ = funcy.merge(converted_args, nonconverted_args)
                logger.debug("Converting using approach '{}'".format(convert.__name__))
                return func(*args_, **kwargs)
            except catch:
                pass
    return wrapped
            
        
def make_robust(transformer):
    transformer.fit = _make_robust_to_tabular_types(transformer.fit, 2)
    # todo optionally catch all errors of transformer and return []
    transformer.transform = _make_robust_to_tabular_types(transformer.transform, 1)
    return transformer 

    
class Feature:
    def __init__(self, input, transformer,
                 name=None, description=None, output=None,
                 options = {}):
        self.input = input
        self.name = name
        self.description = description
        self.output = output
        self.options = options
        
        if funcy.is_seqcont(transformer):
            transformer = make_robust_transformer_pipeline(*transformer)
        self.transformer = make_robust(transformer)
        
        
    def as_sklearn_pandas_tuple(self):
        return (self.input, self.transformer)
    
        
def check(func):
    @funcy.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
            return True
        except AssertionError:
            return False
    wrapped.is_check = True
    return wrapped
        
class FeatureValidator:
    
    @check
    def transformer_interface(self, transformer):
        assert hasattr(transformer, 'fit')
        assert hasattr(transformer, 'transform')
    
    def get_all_checks(self):
        for method in self.__dir__():
            if hasattr(method, 'is_check') and method.is_check:
                name = method.__name__[1:]
                yield (method, name) 
    
    def validate_transformer(self, transformer):
        for check, name in self.get_all_checks():
            pass
        return True
