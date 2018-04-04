import unittest

import numpy as np
import pandas as pd

from dengue_prediction.models.input_type_transforms import (
    FeatureTypeTransformer, TargetTypeTransformer)
from dengue_prediction.tests.util import check_frame_equal, seeded


class TestInputTypeTransforms(unittest.TestCase):
    def test_feature_type_transformer(self):
        with seeded(5):
            x_pd = pd.util.testing.makeDataFrame()
        feature_type_transformer = FeatureTypeTransformer()
        x = feature_type_transformer.fit_transform(x_pd)

        # output should be nx2 np.ndarray
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape, x_pd.shape)

        x_pd_inv = feature_type_transformer.inverse_transform(x)
        # transform back
        self.assertTrue(
            check_frame_equal(x_pd, x_pd_inv)
        )

    def test_target_type_transformer_binary(self):
        y = np.array([0, 1, 0, 1, 0, 0])
        target_type_transformer = TargetTypeTransformer(
            needs_label_binarizer=False)
        y1 = target_type_transformer.fit_transform(y)

        # the output should be 1d
        self.assertEqual(y1.ndim, 1)
        self.assertTrue(
            np.array_equal(y, y1)
        )

    def test_target_type_transformer_multiclass(self):
        y = np.array([0, 1, 2, 0, 1, 2, 2, 1])
        target_type_transformer = TargetTypeTransformer(
            needs_label_binarizer=True)
        y1 = target_type_transformer.fit_transform(y)

        # output should be 2d, with shape[1] == 3
        self.assertEqual(y1.ndim, 2)
        self.assertEqual(y1.shape[1], 3)
        self.assertEqual(y1.shape[0], y.shape[0])

    # TODO check additional dtypes
