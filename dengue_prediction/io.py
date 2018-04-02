import pickle

import h5py
import os
import numpy as np
import pandas as pd

from dengue_prediction.util import splitext2

def _check_ext(ext, expected):
    if ext != expected:
        msg = ('File path has wrong extension: {} (expected {})'
               .format(ext, expected))
        raise ValueError(msg)

def write_tabular(obj, filepath):
    _, fn, ext = splitext2(filepath)
    if ext == '.h5':
        _write_tabular_h5(obj, filepath)
    elif ext == '.pkl':
        _write_tabular_pickle(obj, filepath)
    else:
        raise NotImplementedError

def _write_tabular_pickle(obj, filepath):
    _, fn, ext = splitext2(filepath)
    _check_ext(ext, '.pkl')
    if isinstance(obj, np.ndarray):
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    elif isinstance(obj, pd.core.frame.NDFrame):
        obj.to_pickle(filepath)
    else:
        raise NotImplementedError

def _write_tabular_h5(obj, filepath):
    _, fn, ext = splitext2(filepath)
    _check_ext(ext, '.h5')
    if isinstance(obj, np.ndarray):
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset(fn,  data=obj)
    elif isinstance(obj, pd.core.frame.NDFrame):
        obj.to_hdf(filepath, key=fn)
    else:
        raise NotImplementedError

def read_tabular(filepath):
    _, fn, ext = splitext2(filepath)
    if ext == '.h5':
        return _read_tabular_h5(filepath)
    elif ext == '.pkl':
        return _read_tabular_pickle(filepath)
    else:
        raise NotImplementedError

def _read_tabular_h5(filepath):
    _, fn, ext = splitext2(filepath)
    _check_ext(ext, '.h5')
    with h5py.File(filepath, 'r') as hf:
        dataset = hf[fn]
        data = dataset[:]
        return data

def _read_tabular_pickle(filepath):
    _, fn, ext = splitext2(filepath)
    _check_ext(ext, '.pkl')
    with open(filepath, 'rb') as f:
        return pickle.load(f)
