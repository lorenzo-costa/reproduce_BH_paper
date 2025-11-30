from src.helper_functions.metrics import FalseDiscoveryRate, Power
from src.helper_functions.dgps import NormalGenerator, generate_means, compute_p_values
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count
import os
from src.helper_functions.methods import (
    BenjaminiHochberg,
    Bonferroni,
    BonferroniHochberg,
)


scheme_dict = {
    "D": 1,
    "E": 2,
    "I": 3,
}


def run_single_simulation(args):
    """Run a single simulation iteration.

    Parameters
    ----------
    args : tuple
        (i, m, m0_fraction, L, scheme, method, alpha, metrics, child_seed)
    """
    i, m, m0_fraction, L, scheme, methods, alpha, metrics, child_seed = args

    # Create RNG from the spawned seed sequence
    rng = np.random.default_rng(child_seed)
    results = []
    samples_dict = {}

    for m_i in m:
        samples = NormalGenerator(loc=0, scale=1).generate(m_i, rng=rng)
        samples_dict[m_i] = samples
        # by generating this now it pre-allocates memory and we do not need to
        # do for each scenario

        for m0_fraction_i, L_i, scheme_i in itertools.product(m0_fraction, L, scheme):
            # run one scenario

            m0 = int(m_i * m0_fraction_i)
            means = np.zeros(m_i)
            means[: m_i - m0] = generate_means(
                m=m_i, m0=m0, scheme=scheme_dict[scheme_i], L=L_i
            )
            shifted_samples = samples + means
            p_values = compute_p_values(shifted_samples)

            for method_i in methods:
                scenario_out = {
                    "m": m_i,
                    "m0_fraction": m0_fraction_i,
                    "m0": m0,
                    "L": L_i,
                    "scheme": scheme_i,
                    "method": method_i.name,
                }
                rejected = method_i(p_values, alpha)
                for eval_metric in metrics:
                    scenario_out[eval_metric.name] = eval_metric(rejected, means)
                scenario_out["nsim"] = i + 1
                results.append(scenario_out)

    return results, samples_dict


def run_simulation_parallel(
    m,
    m0_fraction,
    L,
    scheme,
    methods,
    alpha,
    metrics=None,
    nsim=100,
    rng=None,
    results_dir=None,
    n_jobs=None,
    verbose=True,
):
    """Run simulation study in parallel for all combinations of parameters.

    Parameters
    ----------
    m : list or np.ndarray of int
        Number of hypotheses tested
    m0_fraction : list or np.ndarray of float
        Fraction of true null hypotheses
    L : list or np.ndarray of int
        Upper bound on non-zero means
    scheme : list or np.ndarray of str
        Testing scheme to use
    method : list or np.ndarray of MultipleTesting
        Multiple testing correction methods to apply
    alpha : float
        Significance level
    metrics : list, optional
        List of evaluation metrics
    nsim : int, optional
        Number of simulations to run, by default 100
    rng : np.random.Generator, optional
        Random number generator, by default None
    results_dir : str, optional
        Directory to save results, by default "results/"
    n_jobs : int, optional
        Number of parallel jobs. If None, uses all available CPUs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing simulation results for all scenarios
    list
        List of sample dictionaries from each simulation
    """
    if rng is None:
        rng = np.random.default_rng()

    if metrics is None:
        raise ValueError("At least one metric must be provided.")

    if not isinstance(m, (list, np.ndarray)):
        m = [m]
    if not isinstance(m0_fraction, (list, np.ndarray)):
        m0_fraction = [m0_fraction]
    if not isinstance(L, (list, np.ndarray)):
        L = [L]
    if not isinstance(scheme, (list, np.ndarray)):
        scheme = [scheme]
    if not isinstance(methods, (list, np.ndarray)):
        methods = [methods]

    # if n_jobs is None, use all available CPUs
    if n_jobs is None:
        n_jobs = cpu_count()
    chunk_size = max(1, nsim // (n_jobs * 7))

    # ensure reproducible parallel random number generation
    child_seeds = rng.spawn(nsim)

    if results_dir is not None:
        os.makedirs(f"{results_dir}/raw", exist_ok=True)

    total_scenarios = len(m) * len(m0_fraction) * len(L) * len(scheme) * len(methods)

    print(f"\nRunning {nsim} simulations with {total_scenarios} scenarios each")
    print(f"Using {n_jobs} parallel processes")

    sim_args = [
        (i, m, m0_fraction, L, scheme, methods, alpha, metrics, child_seeds[i])
        for i in range(nsim)
    ]

    out = []
    samples_list = []
    save_points = np.unique(np.linspace(1, nsim, min(10, nsim), dtype=int))

    with Pool(processes=n_jobs) as pool:
        with tqdm(total=nsim, desc="Running simulations", disable=not verbose) as pbar:
            for i, (results, samples_dict) in enumerate(
                pool.imap(run_single_simulation, sim_args, chunksize=chunk_size)
            ):
                out.extend(results)
                samples_list.append(samples_dict)
                pbar.update(1)

                if results_dir is not None:
                    if (i + 1) in save_points:
                        pd.DataFrame(out).to_csv(
                            f"{results_dir}/raw/simulation_results_checkpoint_{i}.csv",
                            index=False,
                        )
    out = pd.DataFrame(out)
    return out, samples_list


def run_simulation(
    m,
    m0_fraction,
    L,
    scheme,
    methods,
    alpha,
    metrics=None,
    nsim=100,
    rng=None,
    results_dir=None,
    verbose=True,
    parallel=False,
    n_jobs=None,
):
    """Run simulation study for all combinations of parameters.

    Parameters
    ----------
    m : list or np.ndarray of int
        Number of hypotheses tested
    m0 : list or np.ndarray of float
        Fraction of true null hypotheses
    L : list or np.ndarray of int
        Upper bound on non-zero means
    scheme : list or np.ndarray of str
        Testing scheme to use
    method : list or np.ndarray of MultipleTesting
        Multiple testing correction methods to apply
    alpha : float
        Significance level
    nsim : int, optional
        Number of simulations to run, by default 100
    rng : np.random.Generator, optional
        Random number generator, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing simulation results for all scenarios
    """

    if parallel:
        return run_simulation_parallel(
            m=m,
            m0_fraction=m0_fraction,
            L=L,
            scheme=scheme,
            methods=methods,
            alpha=alpha,
            metrics=metrics,
            nsim=nsim,
            rng=rng,
            results_dir=results_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    if rng is None:
        rng = np.random.default_rng()

    if metrics is None:
        raise ValueError("At least one metric must be provided.")

    if not isinstance(m, (list, np.ndarray)):
        m = [m]
    if not isinstance(m0_fraction, (list, np.ndarray)):
        m0_fraction = [m0_fraction]
    if not isinstance(L, (list, np.ndarray)):
        L = [L]
    if not isinstance(scheme, (list, np.ndarray)):
        scheme = [scheme]
    if not isinstance(methods, (list, np.ndarray)):
        methods = [methods]

    if results_dir is not None:
        os.makedirs(f"{results_dir}/raw", exist_ok=True)

    total_scenarios = len(m) * len(m0_fraction) * len(L) * len(scheme) * len(methods)

    print(f"\nRunning {nsim} simulations with {total_scenarios} scenarios each")

    child_seeds = rng.spawn(nsim)

    out = []
    samples_list = []
    save_points = np.unique(np.linspace(1, nsim, min(10, nsim), dtype=int))
    with tqdm(total=nsim, desc="Running simulations", disable=not verbose) as pbar:
        for i in range(nsim):
            if results_dir is not None:
                if (i + 1) in save_points:
                    pd.DataFrame(out).to_csv(
                        f"{results_dir}/raw/simulation_results_checkpoint_{i}.csv",
                        index=False,
                    )

            sim_out = run_single_simulation(
                (i, m, m0_fraction, L, scheme, methods, alpha, metrics, child_seeds[i])
            )

            out.extend(sim_out[0])
            samples_list.append(sim_out[1])
            pbar.update(1)
            # pbar.refresh()

    out = pd.DataFrame(out)
    return out, samples_list
