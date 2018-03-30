import unittest

import numpy as np
import pandas as pd
import sklearn.preprocessing

from dengue_prediction.features.feature import (Feature,
                                                make_robust,)
from dengue_prediction.util import (asarray2d,
                                    IdentityTransformer,
                                    FragileTransformer,)

class TestFeature(unittest.TestCase):
    def setUp(self):
        self.input = 'foo'
        
        self.transformer = IdentityTransformer()
        
        self.X_ser = pd.util.testing.makeFloatSeries()
        self.X_df  = self.X_ser.to_frame()
        self.X_arr1d = np.asarray(self.X_ser)
        self.X_arr2d = np.asarray(self.X_df)
        self.y_ser = self.X_ser.copy()
        self.y_df  = self.X_df.copy()
        self.y_arr1d = np.asarray(self.y_ser)
        self.y_arr2d = np.asarray(self.y_df)
        
        self.d = {
            'ser' : (self.X_ser, self.y_ser),
            'df' : (self.X_df, self.y_df),
            'arr1d': (self.X_arr1d, self.y_arr1d),
            'arr2d': (self.X_arr2d, self.y_arr2d),
        }
        
    def test_feature_init(self):
        feature = Feature(self.input, self.transformer)
        
    def _test_robust_transformer(self, input_types, bad_input_checks, catches):
        fragile_transformer = FragileTransformer(bad_input_checks, catches)
        robust_transformer = make_robust(FragileTransformer(bad_input_checks, catches))
                                         
        for input_type in input_types:
            X, y = self.d[input_type]
            # fragile transformer raises error
            with self.assertRaises(catches):
                X_fragile = fragile_transformer.fit_transform(X, y)
            # robust transformer does not raise error
            X_robust = robust_transformer.fit_transform(X, y)
            self.assertTrue(
                np.array_equal(
                    asarray2d(X),
                    asarray2d(X_robust)
                )
            )
        
    def test_robust_transformer_ser(self):
        input_types = ('ser',)
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer(input_types, bad_input_checks, catches)
    
    def test_robust_transformer_df(self):
        input_types = ('ser', 'df',)
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
            lambda x: isinstance(x, pd.DataFrame),
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer(input_types, bad_input_checks, catches)
    
    def test_robust_transformer_arr(self):
        input_types = ('ser', 'df', 'arr1d')
        bad_input_checks = (
            lambda x: isinstance(x, pd.Series),
            lambda x: isinstance(x, pd.DataFrame),
            lambda x: isinstance(x, np.ndarray) and x.ndim == 1,
        )
        catches = (ValueError, TypeError)
        self._test_robust_transformer(input_types, bad_input_checks, catches)
        
    def _test_robust_transformer_pipeline(self):
        pass
    
    def test_robust_transformer_pipeline_ser(self):
        pass
    
    def test_robust_transformer_pipeline_df(self):
        pass
    
    def test_robust_transformer_pipeline_arr(self):
        pass
        
    def test_feature_as_sklearn_pandas_tuple(self):
        feature = Feature(self.input, self.transformer)
        tup = feature.as_sklearn_pandas_tuple()
        self.assertIsInstance(tup, tuple)
        self.assertEqual(len(tup), 2)
        
    
class TestFeatureValidator(unittest.TestCase):
    pass