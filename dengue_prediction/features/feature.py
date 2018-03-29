import logging


import funcy
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def make_approaches(funcs, catch):
    if funcs:
        for func in funcs[:-1]:
            yield func, catch 
        yield funcs[-1], ()
        

def _make_robust_to_tabular_type(func, n=2):
    @funcy.wraps(func)
    def wrapped(*args, **kwargs):
        approaches = make_approaches(
            (funcy.identity, pd.DataFrame, np.asarray),
            (ValueError, TypeError) 
        )
        for convert, catch in approaches:
            try:
                converted_args = [convert(args[j]) for j in range(n)]
                original_args = args[n:]
                logger.debug("Converting using approach '{}'".format(convert.__name__))
                return func(*converted_args, *original_args, **kwargs)
            except catch:
                pass
    return wrapped
            
        
def make_robust(transformer):
    transformer.fit = _make_robust_to_tabular_type(transformer.fit, n=2)
    transformer.transform = _make_robust_to_tabular_type(transformer.transform, n=1)
    return transformer 

    
class Feature:
    def __init__(self, input, transformer,
                 name=None, description=None, output=None,
                 options = {}):
        self.input = input
        self.transformer = make_robust(transformer)
        self.name = name
        self.description = description
        self.output = output
        self.options = options
        
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
