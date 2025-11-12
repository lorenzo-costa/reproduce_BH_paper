import cProfile
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

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    
parser = argparse.ArgumentParser()
parser.add_argument("--nsim", type=int, default=None)
parser.add_argument("--parallel", default=1)
args = parser.parse_args()
nsim = args.nsim if args.nsim is not None else cfg["nsim"]
parallel = bool(int(args.parallel))

methods = [method_map[name]() for name in cfg["methods"]]
alpha = cfg["alpha"]
m = cfg["m"]
m0 = cfg["m0"]
metrics = [Power(), TrueRejections(), RejectionsNumber(), FalseDiscoveryRate()]
L = cfg["L"]
scheme = cfg["scheme"]
rng = np.random.default_rng(cfg["rng_seed"])

results_dir = cfg.get("results_dir", "results/")
data_dir = cfg.get("data_dir", "data/")

with cProfile.Profile() as pr:
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
        results_dir=data_dir + "/simulated/",
        parallel=parallel,
    )

pr.dump_stats("profile.stats")