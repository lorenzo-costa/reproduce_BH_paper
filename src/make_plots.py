"""
Script to create plots from simulation results
"""

from src.helper_functions.plot_functions import plot_grid, plot_boxplot, plot_with_bands

import matplotlib.pyplot as plt
import pandas as pd
import yaml

func_map = {
    "plot_with_bands": plot_with_bands,
    "plot_boxplot": plot_boxplot,
}


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
