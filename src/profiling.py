import cProfile
import os
import pstats
from src.helper_functions.simulation_functs import run_simulation
from src.helper_functions.metrics import (
    Power,
    TrueRejections,
    RejectionsNumber,
    FalseDiscoveryRate,
)
from src.helper_functions.methods import (
    Bonferroni,
    BonferroniHochberg,
    BenjaminiHochberg,
)

from src.helper_old.simulation_functs import (run_simulation as run_simulation_old)

from src.helper_old.metrics import (
    Power as PowerOld,
    TrueRejections as TrueRejectionsOld,
    RejectionsNumber as RejectionsNumberOld,
    FalseDiscoveryRate as FalseDiscoveryRateOld,
)
from src.helper_old.methods import (
    Bonferroni as BonferroniOld,
    BonferroniHochberg as BonferroniHochbergOld,
    BenjaminiHochberg as BenjaminiHochbergOld,
)

import pickle
import numpy as np
import yaml
import time
import argparse

method_map = {
    "Bonferroni": Bonferroni,
    "BonferroniHochberg": BonferroniHochberg,
    "BenjaminiHochberg": BenjaminiHochberg,
}

method_map_old = {
    "Bonferroni": BonferroniOld,
    "BonferroniHochberg": BonferroniHochbergOld,
    "BenjaminiHochberg": BenjaminiHochbergOld,
}

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsim", type=int, default=None)
    parser.add_argument("--parallel", default=1)
    args = parser.parse_args()
    # cannot run full 20k because old non-parallel version is too slow
    nsim = args.nsim if args.nsim is not None else 2000 
    parallel = bool(int(args.parallel))

    methods = [method_map[name]() for name in cfg["methods"]]
    methods_old = [method_map_old[name]() for name in cfg["methods"]]

    alpha = cfg["alpha"]
    m = cfg["m"]
    m0 = cfg["m0"]

    metrics = [Power(), TrueRejections(), RejectionsNumber(), FalseDiscoveryRate()]
    metrics_old = [PowerOld(), TrueRejectionsOld(), RejectionsNumberOld(), FalseDiscoveryRateOld()]
    L = cfg["L"]
    scheme = cfg["scheme"]
    rng = np.random.default_rng(cfg["rng_seed"])

    results_dir = cfg.get("results_dir", "results/")
    data_dir = cfg.get("data_dir", "data/")
    
    os.makedirs(results_dir + "profiling/", exist_ok=True)

    with cProfile.Profile() as pr_new:
        sim_out, samples_list = run_simulation(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods,
            alpha=alpha,
            rng=rng,
            metrics=metrics,
            results_dir=None,
            parallel=parallel,
        )
    pr_new.disable()
    pr_new.dump_stats("results/profiling/profile_new.stats")

    with cProfile.Profile() as pr_old:
         sim_out_old, samples_list_old = run_simulation_old(
            nsim=nsim,
            m=m,
            m0_fraction=m0,
            L=L,
            scheme=scheme,
            method=methods_old,
            alpha=alpha,
            rng=rng,
            metrics=metrics_old,
            results_dir=None,
            parallel=parallel,
            old=True
        )
    pr_old.disable()
    pr_old.dump_stats("results/profiling/profile_old.stats")
