import logging
import traceback

import funcy
import numpy as np
import pandas as pd
from sklearn.pipeline import _name_estimators
from sklearn_pandas.pipeline import TransformerPipeline

from dengue_prediction.util import asarray2d, indent

logger = logging.getLogger(__name__)


__all__ = ['Feature', 'FeatureValidator', 'RobustTransformerPipeline',
           'make_robust_transformer_pipeline']


class RobustTransformerPipeline(TransformerPipeline):

    def transform(self, X, **transform_kwargs):
        _transform = make_robust_to_tabular_types(super().transform)
        return _transform(X, **transform_kwargs)


def make_robust_transformer_pipeline(*steps):
    """Construct a RobustTransformerPipeline from the given estimators."""
    return RobustTransformerPipeline(_name_estimators(steps))


def make_conversion_approaches():
    funcs = (funcy.identity, pd.Series, pd.DataFrame, np.asarray, asarray2d)
    catch = (ValueError, TypeError)
    for func in funcs[:-1]:
        yield func, catch
    yield funcs[-1], ()


def make_robust_to_tabular_types(func):
    @funcy.wraps(func)
    def wrapped(X, y=None, **kwargs):
        for convert, catch in make_conversion_approaches():
            try:
                logger.debug(
                    "Converting using approach '{}'".format(convert.__name__))
                if y is not None:
                    return func(convert(X), y=convert(y), **kwargs)
                else:
                    return func(convert(X), **kwargs)
            except catch as e:
                formatted_exc = indent(traceback.format_exc(), n=8)
                logger.debug(
                    "Application subsequently failed with exception '{}'\n\n{}".format(e.__class__.__name__, formatted_exc))
    return wrapped


def make_robust_transformer(transformer):
    transformer.fit = make_robust_to_tabular_types(transformer.fit)
    # todo optionally catch all errors of transformer and return []
    transformer.transform = make_robust_to_tabular_types(transformer.transform)
    return transformer


class Feature:
    def __init__(self, input, transformer,
                 name=None, description=None, output=None,
                 options={}):
        self.input = input
        self.name = name
        self.description = description
        self.output = output
        self.options = options

        if funcy.is_seqcont(transformer):
            transformer = make_robust_transformer_pipeline(*transformer)
        self.transformer = make_robust_transformer(transformer)

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
