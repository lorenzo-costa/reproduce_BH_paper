from src.helper_functions.simulation_functs import run_simulation
import pytest
from src.helper_functions.methods import (
    Bonferroni,
    BonferroniHochberg,
    BenjaminiHochberg,
)
from src.helper_functions.metrics import Power, TrueRejections, RejectionsNumber
import numpy as np
import pandas as pd


def test_run_simulation():
    nsim = 10
    methods = [Bonferroni(), BonferroniHochberg(), BenjaminiHochberg()]
    alpha = 0.05
    m = [4, 8, 16, 32, 64]
    m0 = [3 / 4, 1 / 2, 1 / 4, 0]
    metrics = [Power(), TrueRejections(), RejectionsNumber()]
    L = [5, 10]
    scheme = ["E", "I", "D"]
    rng = np.random.default_rng(42)

    result_sim, normal_samples = run_simulation(
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        nsim=nsim,
        rng=rng,
        metrics=metrics,
    )

    assert np.all((result_sim["Power"] >= 0) & (result_sim["Power"] <= 1))

    assert result_sim.shape[0] == nsim * len(methods) * len(m) * len(m0) * len(L) * len(
        scheme
    )
    assert result_sim.shape[1] == 7 + len(
        metrics
    )  # 7 for the parameters + number of metrics


def test_run_simulation_reproducibility():
    nsim = 5
    methods = [Bonferroni(), BonferroniHochberg(), BenjaminiHochberg()]
    alpha = 0.05
    m = [8]
    m0 = [1 / 2]
    metrics = [Power(), TrueRejections(), RejectionsNumber()]
    L = [5]
    scheme = ["E"]
    rng1 = np.random.default_rng(42)
    result_sim1, normal_samples1 = run_simulation(
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        nsim=nsim,
        rng=rng1,
        metrics=metrics,
    )

    rng2 = np.random.default_rng(42)
    result_sim2, normal_samples2 = run_simulation(
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        nsim=nsim,
        rng=rng2,
        metrics=metrics,
    )

    assert result_sim1.equals(result_sim2)
    for i in range(len(normal_samples1)):
        assert np.array_equal(normal_samples1[i], normal_samples2[i])


# test that for large scale testing power of fdr > power bonferroni
def test_fdr_power_greater_equal_bonferroni():
    nsim = 20
    methods = [Bonferroni(), BenjaminiHochberg()]
    alpha = 0.05
    m = [1000]
    m0 = [1 / 2]
    metrics = [Power()]
    L = [5]
    scheme = ["E"]
    rng = np.random.default_rng(123)

    result_sim, normal_samples = run_simulation(
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        nsim=nsim,
        rng=rng,
        metrics=metrics,
    )

    for i in range(nsim):
        power_bonf = result_sim[result_sim["method"] == "Bonferroni Correction"][
            "Power"
        ].values[i]
        power_fdr = result_sim[result_sim["method"] == "Benjamini-Hochberg Correction"][
            "Power"
        ].values[i]
        assert power_fdr >= power_bonf, (
            f"FDR power {power_fdr} is less than Bonferroni power {power_bonf}"
        )


def test_parallelization_equal_nonparallel():
    nsim = 200
    methods = [Bonferroni(), BonferroniHochberg(), BenjaminiHochberg()]
    alpha = 0.05
    m = [4, 8, 16]
    m0 = [3 / 4, 1 / 2]
    metrics = [Power(), TrueRejections(), RejectionsNumber()]
    L = [5]
    scheme = ["E", "I", "D"]

    rng = np.random.default_rng(42)
    result_sim_parallel, _ = run_simulation(
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        nsim=nsim,
        rng=rng,
        metrics=metrics,
        parallel=True,
    )

    rng = np.random.default_rng(42)
    result_sim_nonparallel, _ = run_simulation(
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        nsim=nsim,
        rng=rng,
        metrics=metrics,
        parallel=False,
    )
    agg_sequential = (
        result_sim_nonparallel.groupby(["method", "m", "m0_fraction", "scheme", "L"])
        .mean()
        .reset_index()
    )
    agg_parallel = (
        result_sim_parallel.groupby(["method", "m", "m0_fraction", "scheme", "L"])
        .mean()
        .reset_index()
    )

    for metric in metrics:
        metric = metric.name
        if not np.allclose(agg_sequential[metric], agg_parallel[metric], atol=1e-4):
            print(
                f"Discrepancy found in metric {metric} between parallel and non-parallel simulation!"
            )
            print(np.max(np.abs(agg_sequential[metric] - agg_parallel[metric])))
            assert False


def test_different_seed_different_results():
    nsim = 200
    methods = [Bonferroni(), BonferroniHochberg(), BenjaminiHochberg()]
    alpha = 0.05
    m = [4, 8, 16]
    m0 = [3 / 4, 1 / 2]
    metrics = [Power(), TrueRejections(), RejectionsNumber()]
    L = [5]
    scheme = ["E", "I", "D"]

    rng = np.random.default_rng(1)
    result_sim_parallel, _ = run_simulation(
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        nsim=nsim,
        rng=rng,
        metrics=metrics,
        parallel=True,
    )

    rng = np.random.default_rng(42)
    result_sim_nonparallel, _ = run_simulation(
        m=m,
        m0_fraction=m0,
        L=L,
        scheme=scheme,
        method=methods,
        alpha=alpha,
        nsim=nsim,
        rng=rng,
        metrics=metrics,
        parallel=False,
    )
    agg_sequential = (
        result_sim_nonparallel.groupby(["method", "m", "m0_fraction", "scheme", "L"])
        .mean()
        .reset_index()
    )
    agg_parallel = (
        result_sim_parallel.groupby(["method", "m", "m0_fraction", "scheme", "L"])
        .mean()
        .reset_index()
    )

    for metric in metrics:
        metric = metric.name
        # now should be different
        if np.allclose(agg_sequential[metric], agg_parallel[metric], atol=1e-4):
            print(
                f"Discrepancy found in metric {metric} between parallel and non-parallel simulation!"
            )
            print(np.max(np.abs(agg_sequential[metric] - agg_parallel[metric])))
            assert False
