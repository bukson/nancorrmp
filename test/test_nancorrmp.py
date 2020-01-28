import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from nancorrmp.nancorrmp import NaNCorrMp


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

