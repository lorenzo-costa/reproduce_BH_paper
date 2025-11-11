import time
import subprocess
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=5, help="Number of runs to average")
parser.add_argument("--nsim", type=int, default=1000, help="Number of simulations per run")
parser.add_argument("--parallel", type=int, default=0, help="Whether to run in parallel (1) or not (0)")
args = parser.parse_args()

n = args.n
nsim = args.nsim
parallel = args.parallel
times = []
for i in range(1, n + 1):
    print(f"Run {i}")
    start = time.time()
    subprocess.run(["python", "-m", "src.run_simulation", "--nsim", str(nsim), "--parallel", str(parallel)])
    end = time.time()
    times.append(end - start)

print(f"Average elapsed time: {sum(times) / len(times):.3f} seconds +/- {np.std(times):.3f} seconds")
