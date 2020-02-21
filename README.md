Multiprocessing correlation calculation for Python
=======================

[![Build Status](https://travis-ci.com/bukson/nancorrmp.svg?branch=master)](https://travis-ci.com/bukson/nancorrmp)

`nancorrmp` is a small module for calculating correlations of big numpy arrays or pandas dataframes with
 NaNs and infs, using multiple cores. Default `numpy.corrcoef` method does not calculate correlations
 with input that contains NaNs and infs and `pandas` method `pandas.DataFrame.corr` is single thread
 only. 
 
 `nancorrmp` utilizes Pearson correlation calculation code from `scipy`, that is based on `numpy` instead
 of `pandas` cythonic backed. The multiprocessing is implemented by python `multiprocessing` module. 
 `nancorrmp` uses `pandas` method of calculating correlations of arrays with NaNs and infs,
 that skips pair of observations when one of them is either Nan or +inf, or -inf. `nancorrmp` also
 can calculate result with p values, similar to `scipy.pearsonr` function.
 
 Benchmarks are showing that with 4 cores, calculating correlation is faster with `nancorrmp` then with `pandas`
 even for 1200x1200 matrix. With 2 cores it is for 2400x2400. `pandas` single processed implementation is faster
 then using single process `nancorrmp` still for 5000x5000 matrix, so it is recommended to use `nancorrmp` with at least
 2 cores.
 
 Table of Content
================

* [Installation](https://github.com/bukson/nancorrmp#installation)

* [Usage](https://github.com/bukson/nancorrmp#usage)

* [Methods](https://github.com/bukson/nancorrmp#nancorrmp-methods)

* [Benchmark](https://github.com/bukson/nancorrmp#benchmark)

* [Test](https://github.com/bukson/nancorrmp#test)

* [License](https://github.com/bukson/nancorrmp#license)

Installation
============

```
pip install nancorrmp
```
Usage
=====
```python
import pandas as pd
import numpy as np
from nancorrmp.nancorrmp import NaNCorrMp
from pandas.testing import assert_frame_equal

np.random.seed(0)
random_dataframe = pd.DataFrame(np.random.rand(100, 100))
corr = NaNCorrMp.calculate(random_dataframe)
corr_pandas = random_dataframe.corr()
assert_frame_equal(corr, corr_pandas)
corr, p_value = NaNCorrMp.calculate_with_p_value(random_dataframe)
```

NaNCorrMp Methods
=================
`nancorrmp` module has one static class named `NaNCorrMp` with 2 public methods and 1 type

**ArrayLike = Union[pd.DataFrame, np.ndarray]**


Type used to unify `pd.DataFrame` and `np.ndarray`. 


**NaNCorrMp.calculate(X: ArrayLike, n_jobs: int = -1, chunks: int = 500) -> ArrayLike**

Calculates correlation matrix using Pearson correlation. `n_jobs` controls number of cores to use
with default -1 which uses all available cores. `chunks` controls how many pairs of arrays are send to
each process, 500 should be suitable for all purposes. 

Returns output as the same type as input, if `X` is `pd.Dataframe` it will return `pd.Dataframe`, if
`X` is `np.ndarray` it will return `np.ndarray`.

```python
import pandas as pd
import numpy as np
from nancorrmp.nancorrmp import NaNCorrMp

np.random.seed(0)
random_dataframe = pd.DataFrame(np.random.rand(100, 100))
corr = NaNCorrMp.calculate(random_dataframe)
```


**NaNCorrMp.calculate_with_p_value(X: ArrayLike, n_jobs: int = -1, chunks: int = 500) -> Tuple[ArrayLike, ArrayLike]**

Calculates correlation matrix and p value matrix using Pearson correlation. `n_jobs` controls number of cores to use
with default -1 which uses all available cores. `chunks` controls how many pairs of arrays are send to
each process, 500 should be suitable for all purposes. Correlation and p value are the same as the result of 
using `scipy.pearsonr`, but it can be used with NaNs and infs and multiple cores.

Returns output as similar type as input, if `X` is `pd.Dataframe` it will return `(pd.Dataframe, pd.Dataframe)`, if
`X` is `np.ndarray` it will return `(np.ndarray, np.ndarray)`.

```python
import pandas as pd
import numpy as np
from nancorrmp.nancorrmp import NaNCorrMp

np.random.seed(0)
random_dataframe = pd.DataFrame(np.random.rand(100, 100))
corr, p_value = NaNCorrMp.calculate_with_p_value(random_dataframe)
```


Benchmark
============

Results can be reproduced by using `test/test_benchmark_nancorrmp.py` module

```python
import pandas as pd
import numpy as np
from nancorrmp.nancorrmp import NaNCorrMp

np.random.seed(0)
random_dataframe = pd.DataFrame(np.random.rand(1200, 1200))

%timeit NaNCorrMp.calculate(random_dataframe, n_jobs=4, chunks=1000)
# 9.92 s ± 205 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit random_dataframe.corr()
# 10.4 s ± 56.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

random_dataframe = pd.DataFrame(np.random.rand(2400, 2400))

%timeit NaNCorrMp.calculate(random_dataframe, n_jobs=2, chunks=1000)
# 1min 26s ± 3.16 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit random_dataframe.corr()
# 1min 45s ± 3.58 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Test
====

`test` module contains test both for single core usage as for multiple cores. Tests asserts
then the outuput of `NaNCorrMp.calculate` is the same as output of `pandas.corr` for the same data. 
Tests require `scipy` and can be run with the following command:
```bash
python setup.py test
```
Licencse
========

MIT License

Copyright (c) 2020 Michał Bukowski michal.bukowski@buksoft.pl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.