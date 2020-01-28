import ctypes
from multiprocessing.sharedctypes import RawArray
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import *

shared_variables_dictionary = {}


class NaNCorrMp:

    @staticmethod
    def _init_worker(X: RawArray, X_finite_mask: RawArray, X_corr: RawArray, X_shape: Tuple[int, int], X_corr_shape: Tuple[int, int]) -> None:
        shared_variables_dictionary['X'] = X
        shared_variables_dictionary['X_finite_mask'] = X_finite_mask
        shared_variables_dictionary['X_corr'] = X_corr
        shared_variables_dictionary['X_shape'] = X_shape
        shared_variables_dictionary['X_corr_shape'] = X_corr_shape

    @staticmethod
    def calculate(X: pd.DataFrame, n_jobs: int = -1, chunks: int = 500) -> pd.DataFrame:
        if n_jobs == 1:
            return NaNCorrMp.calculate_with_single_core(X)

        X_array = X.to_numpy(dtype=np.float64, copy=True).transpose()
        finite_mask_data = np.isfinite(X_array)
        X_corr = np.ndarray(shape=(X_array.shape[0], X_array.shape[0]), dtype=np.float64)

        X_raw = RawArray(ctypes.c_double, X_array.shape[0] * X_array.shape[1])
        finite_mask_raw = RawArray(ctypes.c_bool, X_array.shape[0] * X_array.shape[1])
        X_corr_raw = RawArray(ctypes.c_double, X_corr.shape[0] * X_corr.shape[1])

        X_np = np.frombuffer(X_raw, dtype=np.float64).reshape(X_array.shape)
        finite_mask_np = np.frombuffer(finite_mask_raw, dtype=np.bool).reshape(X_array.shape)
        X_corr_np = np.frombuffer(X_corr_raw, dtype=np.float64).reshape(X_corr.shape)

        np.copyto(X_np, X_array)
        np.copyto(finite_mask_np, finite_mask_data)

        arguments = ((j, i) for i in range(X_array.shape[0]) for j in range(i))
        processes = n_jobs if n_jobs > 0 else None
        with mp.Pool(processes=processes,
                     initializer=NaNCorrMp._init_worker,
                     initargs=(X_raw, finite_mask_raw, X_corr_raw, X_np.shape, X_corr_np.shape)) \
                as pool:
            list(pool.imap_unordered(NaNCorrMp._corr, arguments, chunks))
        for i in range(X_corr_np.shape[0]):
            X_corr_np[i][i] = 1.0
        return pd.DataFrame(X_corr_np, columns=X.columns, index=X.columns)

    @staticmethod
    def calculate_with_single_core(X: pd.DataFrame) -> pd.DataFrame:
        X_array = X.to_numpy(dtype=np.float64).transpose()
        coeffs = np.ndarray(shape=(X_array.shape[0], X_array.shape[0]), dtype=np.float64)
        finite_mask_data = np.isfinite(X_array)
        N = X_array.shape[0]
        for y_i in range(N):
            for x_i in range(y_i):
                finites = finite_mask_data[x_i] & finite_mask_data[y_i]
                x = X_array[x_i][finites]
                y = X_array[y_i][finites]
                mx = x.mean()
                my = y.mean()
                xm, ym = x - mx, y - my
                r_num = np.add.reduce(xm * ym)
                r_den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
                r = r_num / r_den
                corr = max(min(r, 1.0), -1.0)
                coeffs[x_i][y_i] = corr
                coeffs[y_i][x_i] = corr
        for y_i in range(N):
            coeffs[y_i][y_i] = 1.0
        return pd.DataFrame(coeffs, columns=X.columns, index=X.columns)

    @staticmethod
    def _corr(arguments: Tuple[int, int]) -> None:
        j, i = arguments
        X_np = np.frombuffer(shared_variables_dictionary['X'], dtype=np.float64).reshape(shared_variables_dictionary['X_shape'])
        X_corr_np = np.frombuffer(shared_variables_dictionary['X_corr'], dtype=np.float64).reshape(shared_variables_dictionary['X_corr_shape'])
        finite_mask = np.frombuffer(shared_variables_dictionary['X_finite_mask'], dtype=bool).reshape(shared_variables_dictionary['X_shape'])
        finites = finite_mask[i] & finite_mask[j]
        x = X_np[i][finites]
        y = X_np[j][finites]
        mx, my = x.mean(), y.mean()
        xm, ym = x - mx, y - my
        r_num = np.add.reduce(xm * ym)
        r_den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
        r = r_num / r_den
        corr = max(min(r, 1.0), -1.0)
        X_corr_np[i][j] = corr
        X_corr_np[j][i] = corr
