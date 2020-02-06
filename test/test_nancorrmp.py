import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from nancorrmp.nancorrmp import NaNCorrMp
from scipy.stats import pearsonr


class TestNaNCorrMp(unittest.TestCase):
    X = pd.DataFrame({'a': [1, 5, 7, 9, 4], 'b': [-1, 3, -3, 1, 4], 'c': [0, 1, -1, -2, 4], 'd': [-5, -3, -2, 1, 4]})
    X_nans = pd.DataFrame({'a': [float('NaN'), 5, 7, 9], 'b': [-1, 3, float('NaN'), 1], 'c': [0, 1, -1, -2], 'd': [-5, -3, -2, 1]})
    X_infs = pd.DataFrame({'a': [float('-inf'), 5, 7, 9], 'b': [-1, 3, -3, 1], 'c': [0, 1, -1, -2], 'd': [-5, -3, -2, float(('+inf'))]})

    def test_without_nans(self) -> None:
        result = NaNCorrMp.calculate(self.X, n_jobs=2, chunks=1)
        expected_result = self.X.corr()
        assert_frame_equal(result, expected_result)

    def test_with_nans(self) -> None:
        result = NaNCorrMp.calculate(self.X_nans, n_jobs=2, chunks=1)
        expected_result = self.X_nans.corr()
        assert_frame_equal(result, expected_result)

    def test_with_numpy_input(self) -> None:
        result = NaNCorrMp.calculate(self.X_nans.to_numpy().transpose(), n_jobs=2, chunks=1)
        self.assertEqual(type(result), np.ndarray)
        expected_result = self.X_nans.corr()
        assert_array_almost_equal(result, expected_result.to_numpy().transpose())

    def test_with_infs(self) -> None:
        result = NaNCorrMp.calculate(self.X_infs, n_jobs=2, chunks=1)
        expected_result = self.X_infs.corr()
        assert_frame_equal(result, expected_result)

    def test_single_core_without_nans(self) -> None:
        result = NaNCorrMp.calculate(self.X, n_jobs=1)
        expected_result = self.X.corr()
        assert_frame_equal(result, expected_result)

    def test_single_core_with_nans(self) -> None:
        result = NaNCorrMp.calculate(self.X_nans, n_jobs=1)
        expected_result = self.X_nans.corr()
        assert_frame_equal(result, expected_result)

    def test_single_core_with_infs(self) -> None:
        result = NaNCorrMp.calculate(self.X_infs, n_jobs=1, chunks=1)
        expected_result = self.X_infs.corr()
        assert_frame_equal(result, expected_result)

    def test_calculate_with_p_value_without_nans(self) -> None:
        correlations, p_values = NaNCorrMp.calculate_with_p_value(self.X, n_jobs=2, chunks=1)
        empty_dataframe = pd.DataFrame(columns=correlations.columns, index=correlations.index, copy=True, dtype=np.float64)
        expected_correlations, expected_p_values = empty_dataframe.copy(), empty_dataframe.copy()
        for column in self.X.columns:
            for other_column in self.X.columns:
                expected_correlation, expected_p_value = pearsonr(self.X[column], self.X[other_column])
                expected_correlations[column][other_column] = expected_correlation
                expected_p_values[column][other_column] = expected_p_value
        assert_frame_equal(correlations, expected_correlations)
        assert_frame_equal(p_values, expected_p_values)

    def test_calculate_with_p_value_with_nans(self) -> None:
        correlations, p_values = NaNCorrMp.calculate_with_p_value(self.X_nans, n_jobs=2, chunks=1)
        self.assertFalse(correlations.isnull().values.any())
        self.assertFalse(p_values.isnull().values.any())
