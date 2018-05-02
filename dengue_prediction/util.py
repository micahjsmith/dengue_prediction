import inspect
import logging
import os.path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 1754


class InputLogger(BaseEstimator, TransformerMixin):
    def __init__(self, name=None, level='debug'):
        initialized = False
        if isinstance(level, int):
            self.level = level
            initialized = True
        elif isinstance(level, str):
            level = level.upper()
            if hasattr(logging, level):
                self.level = getattr(logging, level)
                initialized = True

        if not initialized:
            raise ValueError('Invalid level: {}'.format(level))

        self.name = name

    def _log(self, msg):
        # Extract current *.py file name
        # Source: https://stackoverflow.com/a/28645157/2514228
        # TODO get a couple frames up in the stack to get accurate logger name
        if self.name:
            name = self.name
        else:
            name = (inspect.getfile(inspect.currentframe()).split(
                "\\", -1)[-1]).rsplit(".", 1)[0]
        return logging.getLogger(name).log(self.level, msg)

    def fit(self, X, y=None, **fit_kwargs):
        X_desc = get_arr_desc(X)
        y_desc = get_arr_desc(y)
        self._log('Fit called with X={X_desc}, y={y_desc}'.format(
            X_desc=X_desc, y_desc=y_desc))
        return self

    def transform(self, X, **transform_kwargs):
        X_desc = get_arr_desc(X)
        self._log('Transform called with X={X_desc}'.format(X_desc=X_desc))
        return X


def indent(text, n=4):
    _indent = ' ' * n
    return '\n'.join([_indent + line for line in text.split('\n')])


def str_to_enum_member(s, E):
    '''Find the member of enum E with name given by string s (ignoring case)'''
    for member in E:
        if member.name == s.upper():
            return member
    return None


def str_to_class_member(s, C):
    '''Find the member of enum E with name given by string s (ignoring case)'''
    for member in dir(C):
        if not (member.startswith('__') and member.endswith('__')):
            if member == s.upper():
                return getattr(C, member)
    return None


def spliceext(filepath, s):
    '''Add s into filepath before the extension'''
    root, ext = os.path.splitext(filepath)
    return root + s + ext


def replaceext(filepath, new_ext):
    root, ext = os.path.splitext(filepath)
    return root + new_ext


def splitext2(filepath):
    '''Split filepath into root, filename, ext'''
    d, fn = os.path.split(filepath)
    fn, ext = os.path.splitext(fn)

    return d, fn, ext
