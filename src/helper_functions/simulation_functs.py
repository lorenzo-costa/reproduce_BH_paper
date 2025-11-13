from src.helper_functions.metrics import FalseDiscoveryRate, Power
from src.helper_functions.dgps import NormalGenerator, generate_means, compute_p_values
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count
import os
from src.helper_functions.methods import BenjaminiHochberg, Bonferroni, BonferroniHochberg


scheme_dict = {
    "D": 1,
    "E": 2,
    "I": 3,
}

def run_scenario(samples, 
                 m0_fraction, 
                 L, 
                 scheme, 
                 method, 
                 alpha, 
                 metrics, 
                 rng=None):
    
    m = samples.shape[0]
    m0 = int(m * m0_fraction)
    # pre-allocate memory
    means = np.zeros(m)
    # numba doesn't like strings so pass an int
    means[:m-m0] = generate_means(m=m, m0=m0, scheme=scheme_dict[scheme], L=L)
    rng.shuffle(means)
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
        # by generating this now it pre-allocates memory and we do not need to 
        # do for each scenario

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

    out = []
    samples_list = []
    save_points = np.unique(np.linspace(1, nsim, min(10, nsim), dtype=int))

    with Pool(processes=n_jobs) as pool:
        with tqdm(total=total_runs, desc="Running simulations") as pbar:
            for i, (results, samples_dict) in enumerate(
                pool.imap(run_single_simulation, sim_args)
            ):
                out.extend(results) 
                samples_list.append(samples_dict)

                pbar.update(len(results))

                if (i + 1) in save_points:
                    pd.DataFrame(out).to_csv(
                        f"{results_dir}/simulation_results_checkpoint_{i}.csv",
                        index=False,
                    )
    out = pd.DataFrame(out)
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
    show_progress=True,
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

    out = []
    samples_list = []
    save_points = np.unique(np.linspace(1, nsim, min(10, nsim), dtype=int))
    with tqdm(total=total_runs, desc="Running simulations", disable=not show_progress) as pbar:
        for i in range(nsim):
            if (i + 1) in save_points:
                pd.DataFrame(out).to_csv(
                    f"{results_dir}/raw/simulation_results_checkpoint_{i}.csv",
                    index=False,
                )

            for m_i in m:
                samples = NormalGenerator(loc=0, scale=1).generate(m_i, rng=rng)
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
                    out.append(scenario_out)
                    pbar.update(1)
                    
    out = pd.DataFrame(out)
    return out, samples_list

# """
# Parallel simulation profiler - drop this into your code to diagnose bottlenecks.
# """
# import time
# import numpy as np
# from contextlib import contextmanager
# from collections import defaultdict


# class ParallelProfiler:
#     """Profile parallel simulations to identify bottlenecks."""
    
#     def __init__(self, n_jobs):
#         self.n_jobs = n_jobs
#         self.timings = []
#         self.phase_timings = defaultdict(list)
#         self.wall_start = None
#         self.wall_end = None
    
#     def start(self):
#         """Start wall clock timer."""
#         self.wall_start = time.time()
    
#     def record_simulation(self, duration, phases=None):
#         """Record timing for a completed simulation."""
#         self.timings.append(duration)
#         if phases:
#             for phase, time_val in phases.items():
#                 self.phase_timings[phase].append(time_val)
    
#     def finish(self):
#         """Finish timing and print report."""
#         self.wall_end = time.time()
#         self.print_report()
    
#     def print_report(self):
#         """Print comprehensive profiling report."""
#         if not self.timings:
#             print("No timing data collected!")
#             return
        
#         wall_time = self.wall_end - self.wall_start
#         total_cpu = sum(self.timings)
        
#         print("\n" + "="*60)
#         print("PARALLEL PROFILING REPORT")
#         print("="*60)
        
#         # Overall metrics
#         print("\n📊 Overall Performance:")
#         print(f"  Wall clock time:     {wall_time:.2f}s")
#         print(f"  Total CPU time:      {total_cpu:.2f}s")
#         print(f"  Number of workers:   {self.n_jobs}")
#         print(f"  Number of tasks:     {len(self.timings)}")
        
#         # Efficiency
#         theoretical_max = total_cpu / self.n_jobs
#         efficiency = (total_cpu / (wall_time * self.n_jobs)) * 100
#         speedup = total_cpu / wall_time
        
#         print(f"\n⚡ Efficiency Metrics:")
#         print(f"  Parallelization efficiency: {efficiency:.1f}%")
#         print(f"  Speedup:                    {speedup:.2f}x")
#         print(f"  Theoretical best wall time: {theoretical_max:.2f}s")
#         print(f"  Overhead:                   {wall_time - theoretical_max:.2f}s "
#               f"({(wall_time/theoretical_max - 1)*100:.1f}%)")
        
#         # Task timing statistics
#         mean_time = np.mean(self.timings)
#         std_time = np.std(self.timings)
#         cv = std_time / mean_time if mean_time > 0 else 0
        
#         print(f"\n⏱️  Task Timing Statistics:")
#         print(f"  Mean ± Std:         {mean_time:.2f}s ± {std_time:.2f}s")
#         print(f"  Min / Max:          {min(self.timings):.2f}s / {max(self.timings):.2f}s")
#         print(f"  Median:             {np.median(self.timings):.2f}s")
#         print(f"  Coefficient of variation: {cv:.1%}")
        
#         # Load imbalance analysis
#         print(f"\n⚖️  Load Balance Analysis:")
#         if cv < 0.1:
#             print(f"  ✅ Excellent balance (CV < 10%)")
#         elif cv < 0.3:
#             print(f"  ✓  Good balance (CV < 30%)")
#         elif cv < 0.5:
#             print(f"  ⚠️  Moderate imbalance (CV < 50%) - consider finer granularity")
#         else:
#             print(f"  ❌ High imbalance (CV > 50%) - significant load imbalance!")
#             print(f"     → Consider parallelizing at a finer level (e.g., scenarios)")
        
#         # Estimate idle time
#         max_time = max(self.timings)
#         idle_fraction = (wall_time - max_time) / wall_time
#         print(f"  Longest task:       {max_time:.2f}s ({max_time/wall_time*100:.1f}% of wall time)")
#         print(f"  Estimated idle:     ~{idle_fraction*100:.1f}% of worker time")
        
#         # Phase breakdown
#         if self.phase_timings:
#             print(f"\n🔍 Phase Breakdown (average per task):")
#             total_phase_time = sum(np.mean(times) for times in self.phase_timings.values())
#             for phase, times in sorted(self.phase_timings.items()):
#                 mean_phase = np.mean(times)
#                 pct = (mean_phase / total_phase_time * 100) if total_phase_time > 0 else 0
#                 print(f"  {phase:.<25} {mean_phase:>6.2f}s ({pct:>5.1f}%)")
        
#         # Recommendations
#         print(f"\n💡 Recommendations:")
#         if efficiency < 70:
#             print(f"  • Low efficiency detected. Possible causes:")
#             print(f"    - Too much overhead (try larger chunksize)")
#             print(f"    - Workers waiting (check for I/O bottlenecks)")
#             print(f"    - Not enough work per task (batch more work)")
#         if cv > 0.3:
#             print(f"  • High load imbalance. Consider:")
#             print(f"    - Using imap_unordered() instead of imap()")
#             print(f"    - Parallelizing at scenario level instead of simulation level")
#             print(f"    - Reducing chunksize for better distribution")
#         if speedup < self.n_jobs * 0.7:
#             print(f"  • Speedup is less than 70% of cores. Check for:")
#             print(f"    - Serialization overhead (use multiprocessing efficiently)")
#             print(f"    - GIL contention (if using threads instead of processes)")
#             print(f"    - I/O bottlenecks (file/network operations)")
        
#         if efficiency > 80 and cv < 0.2:
#             print(f"  ✅ Performance looks good! No major issues detected.")
        
#         print("="*60 + "\n")


# # Usage wrapper for your simulation function
# def run_single_simulation_profiled(args):
#     """Wrapper that profiles individual simulations."""
#     start = time.time()
    
#     # Track phases (customize for your code)
#     phase_times = {}
    
#     # Your existing simulation code here
#     results, samples = run_single_simulation(args)
    
#     total_time = time.time() - start
    
#     return results, samples, total_time, phase_times


# # Modified parallel runner
# def run_simulation_parallel_profiled(
#     m, m0_fraction, L, scheme, method, alpha,
#     metrics=None, nsim=100, rng=None, 
#     results_dir="results/", n_jobs=None
# ):
#     """Your existing function with profiling added."""
    
#     if rng is None:
#         rng = np.random.default_rng()

#     if metrics is None:
#         raise ValueError("At least one metric must be provided.")

#     if not isinstance(m, (list, np.ndarray)):
#         m = [m]
#     if not isinstance(m0_fraction, (list, np.ndarray)):
#         m0_fraction = [m0_fraction]
#     if not isinstance(L, (list, np.ndarray)):
#         L = [L]
#     if not isinstance(scheme, (list, np.ndarray)):
#         scheme = [scheme]
#     if not isinstance(method, (list, np.ndarray)):
#         method = [method]

#     # if n_jobs is None, use all available CPUs
#     if n_jobs is None:
#         n_jobs = cpu_count()

#     # ensure reproducible parallel random number generation
#     child_seeds = rng.spawn(nsim)

#     os.makedirs(f"{results_dir}/raw", exist_ok=True)

#     total_scenarios = len(m) * len(m0_fraction) * len(L) * len(scheme) * len(method)
#     total_runs = nsim * total_scenarios

#     print(f"Running {nsim} simulations with {total_scenarios} scenarios each")
#     print(f"Total runs: {total_runs}")
#     print(f"Using {n_jobs} parallel processes")

#     sim_args = [
#         (i, m, m0_fraction, L, scheme, method, alpha, metrics, child_seeds[i])
#         for i in range(nsim)
#     ]

#     out = []
#     samples_list = []
#     save_points = np.unique(np.linspace(1, nsim, min(10, nsim), dtype=int))
    
#     profiler = ParallelProfiler(n_jobs)
#     profiler.start()
    
#     out = []
#     samples_list = []
    
#     with Pool(processes=n_jobs) as pool:
#         with tqdm(total=total_runs, desc="Running simulations") as pbar:
#             for results, samples_dict, duration, phases in pool.imap(
#                 run_single_simulation_profiled, sim_args
#             ):
#                 profiler.record_simulation(duration, phases)
#                 out.extend(results)
#                 samples_list.append(samples_dict)
#                 pbar.update(len(results))
    
#     profiler.finish()
    
#     out = pd.DataFrame(out)
#     return out, samples_list

# if __name__ == "__main__":
#     # Profile a small run first
#     results, samples = run_simulation_parallel_profiled(
#         m=[4, 8, 16, 32, 64],
#         m0_fraction=[0.75, 0.5, 0.25, 0.0],
#         L=[5, 10],
#         scheme=['E', 'I', 'D'],
#         method=[Bonferroni(), BonferroniHochberg(), BenjaminiHochberg()],
#         alpha=0.05,
#         metrics=[Power(), FalseDiscoveryRate()],
#         nsim=2000,  
#         n_jobs=8
#     )