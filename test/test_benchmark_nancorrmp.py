import unittest
import numpy as np
import pandas as pd
import time
from numpy.testing import assert_array_almost_equal
from numpy.random import randint

from nancorrmp.nancorrmp import NaNCorrMp


class TestBenchmarkNaNCorrMp(unittest.TestCase):

    @unittest.skip("computation to long")
    def test_benchmark_with_pandas_without_nans(self) -> None:
        # faster when size > 1200, n_jobs=4
        size = 1200
        n_jobs = 4
        np.random.seed(0)
        random_dataframe = pd.DataFrame(np.random.rand(size, size))
        t_start = time.perf_counter()
        nancorrmp_result = NaNCorrMp.calculate(random_dataframe, n_jobs=n_jobs, chunks=1000)
        nancorrmp_time = time.perf_counter() - t_start
        print(f'NaNCorrMp time with {n_jobs} jobs: {nancorrmp_time}')
        t_start = time.perf_counter()
        pandas_result = random_dataframe.corr()
        pandas_time = time.perf_counter() - t_start
        print(f'pandas time: {pandas_time}')
        print(f'nancorrmp_time / pandas_time ratio: {nancorrmp_time/pandas_time}')
        assert_array_almost_equal(nancorrmp_result.to_numpy(), pandas_result.to_numpy())
        self.assertTrue(nancorrmp_time < pandas_time)

    @unittest.skip("computation to long")
    def test_benchmark_with_pandas_with_nans(self) -> None:
        # faster when size > 1200, n_jobs=4
        size = 5000
        n_jobs = 4
        np.random.seed(0)
        nan_infs_ratio = 0.03
        random_dataframe = self._get_random_dataframe_with_nans_and_infs(size, nan_infs_ratio)
        t_start = time.perf_counter()
        nancorrmp_result = NaNCorrMp.calculate(random_dataframe, n_jobs=n_jobs, chunks=1000)
        nancorrmp_time = time.perf_counter() - t_start
        print(f'NaNCorrMp time with {n_jobs} jobs: {nancorrmp_time}')
        t_start = time.perf_counter()
        pandas_result = random_dataframe.corr()
        pandas_time = time.perf_counter() - t_start
        print(f'pandas time: {pandas_time}')
        print(f'nancorrmp_time / pandas_time ratio: {nancorrmp_time / pandas_time}')
        assert_array_almost_equal(nancorrmp_result.to_numpy(), pandas_result.to_numpy())
        self.assertTrue(nancorrmp_time < pandas_time)

    @staticmethod
    def _get_random_dataframe_with_nans_and_infs(size: int, nan_infs_ratio: float) -> pd.DataFrame:
        possible_values = (float('NaN'), float('+inf'), float('-inf'))
        random_dataframe = pd.DataFrame(np.random.rand(size, size))
        for i in range(int(nan_infs_ratio * size * (size - 1) / 2)):
            value = possible_values[i % 3]
            x, y = randint(size), randint(size)
            random_dataframe[x][y] = value
        return random_dataframe

    @unittest.skip("computation to long")
    def test_benchmark_with_numpy_without_nans(self) -> None:
        size = 1200
        n_jobs = 4
        np.random.seed(0)
        random_dataframe = pd.DataFrame(np.random.rand(size, size))
        t_start = time.perf_counter()
        nancorrmp_result = NaNCorrMp.calculate(random_dataframe, n_jobs=n_jobs, chunks=1000)
        nancorrmp_time = time.perf_counter() - t_start
        print(f'NaNCorrMp time with {n_jobs} jobs: {nancorrmp_time}')
        t_start = time.perf_counter()
        numpy_result = np.corrcoef(random_dataframe.to_numpy().transpose())
        numpy_time = time.perf_counter() - t_start
        print(f'numpy time: {numpy_time}')
        print(f'nancorrmp_time / numpy_time ratio: {nancorrmp_time/numpy_time}')
        assert_array_almost_equal(nancorrmp_result.to_numpy(), numpy_result)
        self.assertTrue(nancorrmp_time < numpy_time)
