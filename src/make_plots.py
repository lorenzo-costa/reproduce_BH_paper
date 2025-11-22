"""
Script to create plots from simulation results
"""

from src.helper_functions.plot_functions import plot_grid, plot_boxplot, plot_with_bands

import matplotlib.pyplot as plt
import pandas as pd
import yaml
import pickle
import os
import numpy as np
from scipy.stats import linregress

func_map = {
    "plot_with_bands": plot_with_bands,
    "plot_boxplot": plot_boxplot,
}

def fit_empirical_complexity(n_sim_list, time_mean):
        log_n = np.log(n_sim_list)
        log_time = np.log(time_mean)
        slope, intercept, *_ = linregress(log_n, log_time)
        fitted = np.exp(intercept) * np.power(n_sim_list, slope)
        return slope, intercept, fitted

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    output_path = cfg["figures_dir"]

    plt.rcParams.update(cfg["rcparams"])

    colors = cfg["line_colors"]
    linestyles = cfg["linestyles"]
    name_conversion = cfg["name_conversion"]
    plots = cfg["plots"]

    print("Generating plots...")

    for plot in plots:
        grouped_stats = pd.read_csv(plot["data_dir"])
        plot_name = plot["name"]
        plot_func = func_map[plot["func"]]
        x_axis = plot["x_axis"]
        y_axis = plot["y_axis"]
        factors = plot["factors"]
        height = plot.get("height", 1.3)
        n_boxplots = plot.get("n_boxplots", None)
        se_bands = plot.get("se_bands", None)
        group_variables = plot.get("group_variables", False)
        ratio_variable = plot.get("ratio_variable", None)
        title = plot.get("title", None)

        plot_grid(
            grouped_stats=grouped_stats,
            plotting_function=plot_func,
            x_axis=x_axis,
            y_axis=y_axis,
            factors=factors,
            se_bands=se_bands,
            height=height,
            log_y_axis=False,
            log_x_axis=False,
            group_variables=group_variables,
            n_boxplots=n_boxplots,
            ratio_variable=ratio_variable,
            title=title,
            save_path=output_path + plot_name,
            colors=colors,
            linestyles=linestyles,
            name_conversion=name_conversion,
        )
        print(f"Plot {plot_name} saved.")
    
    # plot timing comparison
    timing_dir = "results/comparison"
    timing_results = []
    
    for root, _, files in os.walk(timing_dir):
        for fname in files:
            if fname.lower().endswith(".pkl"):
                path = os.path.join(root, fname)
                with open(path, "rb") as fh:
                    try:
                        timing_results.append(pickle.load(fh))
                    except Exception as e:
                        print(f"Failed to load {path}: {e}")

    all_results = sorted(timing_results, key=lambda x: x['n_sim_list'])
    
    time_old = [all_results[i]['times_old'] for i in range(len(all_results))]
    time_old_mean = np.mean(time_old, axis=1)
    time_new = [all_results[i]['times_new'] for i in range(len(all_results))]
    time_new_mean = np.mean(time_new, axis=1)
    time_old_concat = [all_results[i]['times_old_concat'] for i in range(len(all_results))]
    time_old_concat_mean = np.mean(time_old_concat, axis=1)
    time_new_parallel = [all_results[i]['times_new_parallel'] for i in range(len(all_results))]
    time_new_parallel_mean = np.mean(time_new_parallel, axis=1)
    n_sim_list = [all_results[i]['n_sim_list'] for i in range(len(all_results))]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_sim_list, time_old_mean, marker='o', label='Old Method')
    ax.plot(n_sim_list, time_new_mean, marker='o', label='New Method')
    ax.plot(n_sim_list, time_old_concat_mean, marker='o', label='Old Method (Concat)')
    ax.plot(n_sim_list, time_new_parallel_mean, marker='o', label='New Method (Parallel)')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Simulations")
    ax.set_ylabel("Time (seconds)")
    ax.legend()
    plt.title("Simulation Timing Comparison")

    if output_path is not None:
        plt.savefig(output_path + "timing.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_path + "timing.pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    print("Timing plot saved.")
    
    # plot empirical complexity fit
    slope_new, intercept_new, fitted_new = fit_empirical_complexity(n_sim_list, time_new_mean)
    fitted_new = np.exp(intercept_new) * np.power(n_sim_list, slope_new)

    # using complexity from BASELINE.md
    # $O\left(n_{sim} \cdot n_{scenarios} \cdot \sum m_{i}(\log m_{i}+K)\right)$
    m = np.array([4, 8, 16, 32, 64])
    k = ((m * np.log(m+3)).sum() * 72)
    theoretical_complexity = np.array(n_sim_list) * k

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(n_sim_list, time_new_mean, 'o-', label='New Method')
    ax.plot(n_sim_list, fitted_new, '--', label=f'Empirical New O(n^{slope_new:.2f})')
    # scale theoretical to be in same scale as data
    ax.plot(n_sim_list, theoretical_complexity * (time_new_mean[0] / (n_sim_list[0] * k)), '--', label='Theoretical Complexity')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('n (log scale)')
    ax.set_ylabel('Runtime (log scale)')
    ax.set_title('Empirical Complexity: Runtime vs n (log-log)')
    ax.legend()
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path + "complexity_analysis.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_path + "complexity_analysis.pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()