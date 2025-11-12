from src.helper_functions.dgps import NormalGenerator, generate_means, compute_p_values
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm


import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os


def run_scenario(samples, m0_fraction, L, scheme, method, alpha, metrics, rng=None):
    m = samples.shape[0]
    m0 = int(m * m0_fraction)
    means = generate_means(m=m, m0=m0, scheme=scheme, L=L, rng=rng)
    # uses property of Gaussian X ~ N(mu, 1) => X = mu + Z, Z ~ N(0,1)
    shifted_samples = samples + means
    p_values = compute_p_values(shifted_samples)
    rejected = method(p_values, alpha)

    results = {
        "m": m,
        "m0_fraction": m0_fraction,
        "m0": m0,
        "L": L,
        "scheme": scheme,
        "method": method.name,
    }

    for eval_metric in metrics:
        results[eval_metric.name] = eval_metric(rejected, means)

    return results


def run_single_simulation(args):
    """Run a single simulation iteration.

    Parameters
    ----------
    args : tuple
        (i, m, m0_fraction, L, scheme, method, alpha, metrics, child_seed)
    """
    i, m, m0_fraction, L, scheme, method, alpha, metrics, child_seed = args

    # Create RNG from the spawned seed sequence
    rng = np.random.default_rng(child_seed)
    results = []
    samples_dict = {}

    for m_i in m:
        samples = NormalGenerator(loc=0, scale=1).generate(m_i, rng=rng)
        samples_dict[m_i] = samples

        for m0_i, L_i, scheme_i, method_i in itertools.product(
            m0_fraction, L, scheme, method
        ):
            scenario_out = run_scenario(
                samples=samples,
                m0_fraction=m0_i,
                L=L_i,
                scheme=scheme_i,
                method=method_i,
                alpha=alpha,
                metrics=metrics,
                rng=rng,
            )
            scenario_out["nsim"] = i + 1
            results.append(scenario_out)

    return results, samples_dict


def run_simulation_parallel(
    m,
    m0_fraction,
    L,
    scheme,
    method,
    alpha,
    metrics=None,
    nsim=100,
    rng=None,
    results_dir="results/",
    n_jobs=None,
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
    if not isinstance(method, (list, np.ndarray)):
        method = [method]

    # if n_jobs is None, use all available CPUs
    if n_jobs is None:
        n_jobs = cpu_count()

    # ensure reproducible parallel random number generation
    child_seeds = rng.spawn(nsim)

    os.makedirs(f"{results_dir}/raw", exist_ok=True)

    total_scenarios = len(m) * len(m0_fraction) * len(L) * len(scheme) * len(method)
    total_runs = nsim * total_scenarios

    print(f"Running {nsim} simulations with {total_scenarios} scenarios each")
    print(f"Total runs: {total_runs}")
    print(f"Using {n_jobs} parallel processes")

    sim_args = [
        (i, m, m0_fraction, L, scheme, method, alpha, metrics, child_seeds[i])
        for i in range(nsim)
    ]

    out = pd.DataFrame()
    samples_list = []
    save_points = np.unique(np.linspace(1, nsim, min(10, nsim), dtype=int))

    with Pool(processes=n_jobs) as pool:
        # imap maintains order and enable progress tracking
        with tqdm(total=total_runs, desc="Running simulations") as pbar:
            for i, (results, samples_dict) in enumerate(
                pool.imap(run_single_simulation, sim_args)
            ):
                out = pd.concat([out, pd.DataFrame(results)], ignore_index=True)
                samples_list.append(samples_dict)

                pbar.update(len(results))

                if (i + 1) in save_points:
                    out.to_csv(
                        f"{results_dir}/simulation_results_checkpoint_{i}.csv",
                        index=False,
                    )

    return out, samples_list


def run_simulation(
    m,
    m0_fraction,
    L,
    scheme,
    method,
    alpha,
    metrics=None,
    nsim=100,
    rng=None,
    results_dir="results/",
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
            m,
            m0_fraction,
            L,
            scheme,
            method,
            alpha,
            metrics,
            nsim,
            rng,
            results_dir,
            n_jobs,
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
    if not isinstance(method, (list, np.ndarray)):
        method = [method]

    total_scenarios = len(m) * len(m0_fraction) * len(L) * len(scheme) * len(method)
    total_runs = nsim * total_scenarios

    out = pd.DataFrame()
    samples_list = []
    save_points = np.unique(np.linspace(1, nsim, min(10, nsim), dtype=int))
    # with tqdm(total=total_runs, desc="Running simulations") as pbar:
    for i in range(nsim):
        if (i + 1) in save_points:
            out.to_csv(
                f"{results_dir}/raw/simulation_results_checkpoint_{i}.csv",
                index=False,
            )

        for m_i in m:
            samples = NormalGenerator(loc=0, scale=1).generate(m_i, rng=rng)
            # TODO: handle this better to speed up code
            samples_list.append(samples)
            for m0_i, L_i, scheme_i, method_i in itertools.product(
                m0_fraction, L, scheme, method
            ):
                scenario_out = run_scenario(
                    samples=samples,
                    m0_fraction=m0_i,
                    L=L_i,
                    scheme=scheme_i,
                    method=method_i,
                    alpha=alpha,
                    metrics=metrics,
                    rng=rng,
                )
                scenario_out["nsim"] = i + 1
                # TODO: Optimize this concatenation
                # this creates a monstrous bottleneck, luckyly the parallel version avoids it
                # may easily get a 50x speedup by gettign this right.
                out = pd.concat(
                    [out, pd.DataFrame(scenario_out, index=[0])], ignore_index=True
                )
                # pbar.update(1)

    return out, samples_list
