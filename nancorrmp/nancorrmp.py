import ctypes
from multiprocessing.sharedctypes import RawArray
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import *
from scipy.special import btdtr

shared_variables_dictionary = {}

ArrayLike = Union[pd.DataFrame, np.ndarray]


class NaNCorrMp:

    @staticmethod
    def _init_worker(X: RawArray, X_finite_mask: RawArray, X_corr: RawArray,
                     X_shape: Tuple[int, int], X_corr_shape: Tuple[int, int],
                     X_p_value: RawArray = None) -> None:
        shared_variables_dictionary['X'] = X
        shared_variables_dictionary['X_finite_mask'] = X_finite_mask
        shared_variables_dictionary['X_corr'] = X_corr
        shared_variables_dictionary['X_shape'] = X_shape
        shared_variables_dictionary['X_corr_shape'] = X_corr_shape
        shared_variables_dictionary['X_p_value'] = X_p_value
        shared_variables_dictionary['X_p_value_shape'] = X_corr_shape

    @staticmethod
    def calculate(X: ArrayLike, n_jobs: int = -1, chunks: int = 500) -> ArrayLike:
        return NaNCorrMp._calculate(X=X, n_jobs=n_jobs, chunks=chunks, add_p_values=False)

    @staticmethod
    def calculate_with_p_value(X: ArrayLike, n_jobs: int = -1, chunks: int = 500) -> Tuple[ArrayLike, ArrayLike]:
        return NaNCorrMp._calculate(X=X, n_jobs=n_jobs, chunks=chunks, add_p_values=True)

    @staticmethod
    def _calculate(X: ArrayLike, n_jobs: int, chunks: int, add_p_values: int) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        X_array = X.to_numpy(dtype=np.float64, copy=True).transpose() if type(X) == pd.DataFrame else X
        X_raw = RawArray(ctypes.c_double, X_array.shape[0] * X_array.shape[1])
        X_np = np.frombuffer(X_raw, dtype=np.float64).reshape(X_array.shape)
        np.copyto(X_np, X_array)

        finite_mask_data = np.isfinite(X_array)
        finite_mask_raw = RawArray(ctypes.c_bool, X_array.shape[0] * X_array.shape[1])
        finite_mask_np = np.frombuffer(finite_mask_raw, dtype=bool).reshape(X_array.shape)
        np.copyto(finite_mask_np, finite_mask_data)

        X_corr = np.ndarray(shape=(X_array.shape[0], X_array.shape[0]), dtype=np.float64)
        X_corr_raw = RawArray(ctypes.c_double, X_corr.shape[0] * X_corr.shape[1])
        X_corr_np = np.frombuffer(X_corr_raw, dtype=np.float64).reshape(X_corr.shape)

        if add_p_values:
            X_p_value = np.ndarray(shape=X_corr.shape, dtype=np.float64)
            X_p_value_raw = RawArray(ctypes.c_double, X_p_value.shape[0] * X_p_value.shape[1])
            X_p_value_np = np.frombuffer(X_p_value_raw, dtype=np.float64).reshape(X_corr.shape)
        else:
            X_p_value_np = None
            X_p_value_raw = None
            X_p_value_np = None

        arguments = ((j, i) for i in range(X_array.shape[0]) for j in range(i))
        processes = n_jobs if n_jobs > 0 else None
        worker_function = NaNCorrMp._set_correlation_with_p_value if add_p_values else NaNCorrMp._set_correlation
        with mp.Pool(processes=processes,
                     initializer=NaNCorrMp._init_worker,
                     initargs=(X_raw, finite_mask_raw, X_corr_raw, X_np.shape, X_corr_np.shape, X_p_value_raw)) \
                as pool:
            list(pool.imap_unordered(worker_function, arguments, chunks))

        for i in range(X_corr_np.shape[0]):
            X_corr_np[i][i] = 1.0

        if add_p_values:
            if type(X) == pd.DataFrame:
                return (
                    pd.DataFrame(X_corr_np, columns=X.columns, index=X.columns),
                    pd.DataFrame(X_p_value_np, columns=X.columns, index=X.columns)
                )
            else:
                return X_corr_np, X_p_value_np

        if type(X) == pd.DataFrame:
            return pd.DataFrame(X_corr_np, columns=X.columns, index=X.columns)
        else:
            return X_corr_np

    @staticmethod
    def _set_correlation(arguments: Tuple[int, int]) -> None:
        j, i = arguments
        X_np, X_corr_np, finite_mask = NaNCorrMp._get_global_variables()
        finites = finite_mask[i] & finite_mask[j]
        x = X_np[i][finites]
        y = X_np[j][finites]
        corr = NaNCorrMp._corr(x, y)
        X_corr_np[i][j] = corr
        X_corr_np[j][i] = corr

    @staticmethod
    def _get_global_variables(get_p_value: bool = False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                                  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        X_np = np.frombuffer(shared_variables_dictionary['X'], dtype=np.float64).reshape(shared_variables_dictionary['X_shape'])
        X_corr_np = np.frombuffer(shared_variables_dictionary['X_corr'], dtype=np.float64).reshape(shared_variables_dictionary['X_corr_shape'])
        finite_mask = np.frombuffer(shared_variables_dictionary['X_finite_mask'], dtype=bool).reshape(shared_variables_dictionary['X_shape'])
        if not get_p_value:
            return X_np, X_corr_np, finite_mask
        else:
            X_p_value_np = np.frombuffer(shared_variables_dictionary['X_p_value'], dtype=np.float64).reshape(shared_variables_dictionary['X_p_value_shape'])
            return X_np, X_corr_np, finite_mask, X_p_value_np

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        mx, my = x.mean(), y.mean()
        xm, ym = x - mx, y - my
        r_num = np.add.reduce(xm * ym)
        r_den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
        r = r_num / r_den
        return max(min(r, 1.0), -1.0)

    @staticmethod
    def _set_correlation_with_p_value(arguments: Tuple[int, int]) -> None:
        j, i = arguments
        X_np, X_corr_np, finite_mask, X_p_value_np = NaNCorrMp._get_global_variables(get_p_value=True)
        finites = finite_mask[i] & finite_mask[j]
        x = X_np[i][finites]
        y = X_np[j][finites]
        corr = NaNCorrMp._corr(x, y)
        X_corr_np[i][j] = corr
        X_corr_np[j][i] = corr
        p_value = NaNCorrMp._p_value(corr, len(x))
        X_p_value_np[i][j] = p_value
        X_p_value_np[j][i] = p_value

    @staticmethod
    def _p_value(corr: float, observation_length: int) -> float:
        ab = observation_length / 2 - 1
        if ab == 0:
            p_value = 1.0
        else:
            p_value = 2 * btdtr(ab, ab, 0.5 * (1 - abs(np.float64(corr))))
        return p_value
