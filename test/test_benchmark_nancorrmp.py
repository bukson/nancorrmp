import unittest
import numpy as np
import pandas as pd
import time

from numpy.testing import assert_array_almost_equal

from nancorrmp.nancorrmp import NaNCorrMp


class TestBenchmarkNaNCorrMp(unittest.TestCase):

    @unittest.skip("computation to long")
    def test_speed_without_nans(self) -> None:
        # faster when size > 1200, n_jobs=4
        size = 1200
        n_jobs = 4
        np.random.seed(0)
        random_dataframe = pd.DataFrame(np.random.rand(size, size))
        t_start = time.perf_counter()
        result = NaNCorrMp.calculate(random_dataframe, n_jobs=n_jobs, chunks=1000)
        nancorrmp_time = time.perf_counter() - t_start
        t_start = time.perf_counter()
        expected_result = random_dataframe.corr()
        pandas_time = time.perf_counter() - t_start
        print(nancorrmp_time)
        print(pandas_time)
        print(nancorrmp_time / pandas_time)
        assert_array_almost_equal(result.to_numpy(), expected_result.to_numpy().transpose())
        self.assertTrue(nancorrmp_time < pandas_time)
