import unittest

import numpy as np
import pandas as pd
import sklearn.preprocessing

from dengue_prediction.features.feature import (Feature,
                                                make_robust,)

class TestFeature(unittest.TestCase):
    def setUp(self):
        self.X_ser = pd.util.testing.makeFloatSeries()
        self.X_df  = self.X_ser.to_frame()
        self.X_arr = self.X_df.as_matrix()
        self.y_ser = self.X_ser.copy()
        self.y_df  = self.X_df.copy()
        self.y_arr = self.X_arr.copy()
        
        self.input = 'foo'
        self.transformer = sklearn.preprocessing.StandardScaler()
        
    def test_feature_init(self):
        feature = Feature(self.input, self.transformer)
        
    def test_transformer(self):
        transformer = self.transformer
        with self.assertRaises(ValueError):
            out_ser = transformer.fit_transform(self.X_ser, self.y_ser)
        out_df = transformer.fit_transform(self.X_df, self.y_df)
        out_arr = transformer.fit_transform(self.X_arr, self.y_arr)
        # todo make sure out_* are equivalent
        
    def test_robust_transformer(self):
        transformer = make_robust(self.transformer)
        out_ser = transformer.fit_transform(self.X_ser, self.y_ser)
        out_df = transformer.fit_transform(self.X_df, self.y_df)
        out_arr = transformer.fit_transform(self.X_arr, self.y_arr)
        # todo make sure out_* are equivalent
        
    def test_feature_as_sklearn_pandas_tuple(self):
        feature = Feature(self.input, self.transformer)
        tup = feature.as_sklearn_pandas_tuple()
        self.assertIsInstance(tup, tuple)
        self.assertEqual(len(tup), 2)
        
    
class TestFeatureValidator(unittest.TestCase):
    pass