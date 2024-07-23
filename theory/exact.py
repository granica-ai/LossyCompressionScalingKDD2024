"""Exact expected error from Lemma 3.2"""

import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import root_scalar

from theory.empirical import Expt


def f(x: float, _n: int, _m: int, expt: Expt, scale_by_n: bool = False) -> float:
    """Equation 15"""
    sum_part = sum([expt.p**-l * expt.q**l / (expt.p**-l + x) for l in range(_m)])
    if scale_by_n:
        kappa = np.log(expt.p) / np.log(expt.q)
        exponent = (kappa + 1) / (2 * kappa + 1)
        return sum_part - _n + _n**exponent * expt.lambda_ / x
    return sum_part - _n + expt.lambda_ / x


def D(_m: int, lambda_star: float, expt: Expt) -> float:
    """Equation 18"""
    return sum(
        [
            expt.p ** (-2 * l) * expt.q**l / (expt.p**-l + lambda_star) ** 2
            for l in range(_m)
        ]
    )


def B(_m: int, _n: int, lambda_star: float, expt: Expt) -> float:
    """Equation 16"""
    return (
        _n
        * lambda_star**2
        / (_n - D(_m, lambda_star, expt))
        * sum(
            expt.p**-l * expt.r**l / (expt.p**-l + lambda_star) ** 2 for l in range(_m)
        )
    )


def V(_m: int, _n: int, lambda_star: float, expt: Expt) -> float:
    """Equation 17"""
    return D(_m, lambda_star, expt) / (_n - D(_m, lambda_star, expt))


def test_err_pred(
    n: int,
    m: int,
    expt: Expt,
    b_const: float = 0,
    v_const: float = 0,
    scale_by_n: bool = False,
) -> float:
    """Lemma 3.2 test error prediction"""
    result = root_scalar(f, bracket=[0.0001, 100], args=(n, m, expt, scale_by_n))
    lambda_star = result.root
    kbar_rpm = expt.r / (expt.p - expt.r) * (expt.r / expt.p) ** (m - 1)
    return (
        expt.tau**2
        + (1 + b_const * n**-0.49) * B(m, n, lambda_star, expt)
        + (expt.tau**2 + kbar_rpm)
        * (1 + v_const * n**-0.99)
        * V(m, n, lambda_star, expt)
        + kbar_rpm
    )


def calc_exact_df(
    expt: Expt,
    ns: np.ndarray = None,
    ms: np.ndarray = None,
    b_const: float = 0,
    v_const: float = 0,
    lambda_options: np.ndarray = None,
    scale_by_n: bool = False,
) -> pd.DataFrame:
    """Calculate the exact expected error for a range of n and m."""
    if ns is None:
        ns = np.linspace(400, 800, 6, dtype=int)
    if ms is None:
        ms = np.arange(2, 7)
    if lambda_options is None:
        lambda_options = [expt.lambda_]

    results = []
    for n, m, lambda_ in product(ns, ms, lambda_options):
        expt.lambda_ = lambda_
        err_pred = test_err_pred(
            n, m, expt, b_const=b_const, v_const=v_const, scale_by_n=scale_by_n
        )
        results.append([n, sum([expt.q**l for l in range(m)]), lambda_, err_pred])
    df = pd.DataFrame(
        results, columns=["n", "L", "lambda", "test_err_pred"]
    ).sort_values("test_err_pred")
    return df.groupby(["n", "L"]).first().reset_index()
