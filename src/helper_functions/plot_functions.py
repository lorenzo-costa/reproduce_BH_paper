import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import logging
import re


# Suppress weird matplotlib category warning for boxplots
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)


def plot_with_bands(x_axis, y_axis, **kwargs):
    """Plot lines with confidence/error bands for each method.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot.
    x_axis : str
        The name of the column to be used for the x-axis.
    y_axis : str
        The name of the column to be used for the y-axis.
    factors : list, optional
        A list of column names to be used as additional factors for grouping,
        by default None
    plot_bands : str, optional
        Name of the column containing the standard error for the y-axis values,
        if None no bands are drawn, by default None
    colors : dict, optional
        A dictionary mapping factor values to colors, by default None
    linestyles : dict, optional
        A dictionary mapping factor values to linestyles, by default None
    """
    data = kwargs.pop("data")
    factors = kwargs.pop("factors", None)
    se_bands = kwargs.pop("se_bands", None)
    colors = kwargs.pop("colors", None)
    linestyles = kwargs.pop("linestyles", None)

    ax = plt.gca()
    hue_variable = factors[0] if factors is not None and len(factors) >= 1 else None

    if hue_variable is not None:
        for hue_var in data[hue_variable].unique():
            subset = data[data[hue_variable] == hue_var].sort_values(x_axis)
            line = ax.plot(
                subset[x_axis],
                subset[y_axis],
                marker="o",
                linestyle=linestyles[hue_var] if linestyles is not None else "-",
                color=colors[hue_var] if colors is not None else None,
                label=hue_var,
            )
            color = line[0].get_color()

            if se_bands is not None:
                ax.fill_between(
                    subset[x_axis],
                    subset[y_axis] - subset[se_bands],
                    subset[y_axis] + subset[se_bands],
                    alpha=0.2,
                    color=color,
                )
    else:
        # assume single line
        subset = data.sort_values(x_axis)
        line = ax.plot(
            subset[x_axis], subset[y_axis], marker="o", linestyle="-", label=None
        )
        color = line[0].get_color()

        if se_bands is not None:
            ax.fill_between(
                subset[x_axis],
                subset[y_axis] - subset[se_bands],
                subset[y_axis] + subset[se_bands],
                alpha=0.2,
                color=color,
            )


def plot_boxplot(x_axis, y_axis, **kwargs):
    """Create a boxplot for the given x and y axes.

    Parameters
    ----------
    x_axis : str
        The name of the column to be used for the x-axis.
    y_axis : str
        The name of the column to be used for the y-axis.
    data : pd.DataFrame
        DataFrame containing the data to plot.
    factors : list, optional
        A list of column names to be used as additional factors for grouping,
        by default None
    log_y_axis : bool, optional
        Whether to use a logarithmic scale for the y-axis, by default False
    log_x_axis : bool, optional
        Whether to use a logarithmic scale for the x-axis, by default False
    n_boxplots : int, optional
        Number of boxplots to display along the x-axis, by default 5
    colors : dict, optional
        A dictionary mapping factor values to colors, by default None
    """
    data = kwargs.pop("data")
    factors = kwargs.pop("factors", None)
    log_y_axis = kwargs.pop("log_y_axis", False)
    log_x_axis = kwargs.pop("log_x_axis", False)
    n_boxplots = kwargs.pop("n_boxplots", 5)
    colors = kwargs.pop("colors", None)

    ax = plt.gca()

    temp = data.copy()

    if log_x_axis is True:
        temp[x_axis] = np.log10(temp[x_axis])
    if log_y_axis is True:
        temp[y_axis] = np.log10(temp[y_axis])

    if n_boxplots < len(temp[x_axis].unique()):
        # Select n_boxplots evenly spaced along x
        df_values = sorted(temp[x_axis].unique())
        selected_dfs = np.linspace(0, len(df_values) - 1, n_boxplots, dtype=int)
        selected_dfs = [df_values[i] for i in selected_dfs]
        temp = temp[temp[x_axis].isin(selected_dfs)]

    hue_variable = None
    if factors is not None and len(factors) >= 1:
        hue_variable = factors[0]

    if hue_variable is not None:
        # Create palette from colors dict if provided
        palette = None
        if colors is not None:
            # Get unique hue values in the data
            hue_order = sorted(temp[hue_variable].unique())
            palette = [colors.get(hue_val, None) for hue_val in hue_order]

        sns.boxplot(
            data=temp, x=x_axis, y=y_axis, hue=hue_variable, palette=palette, ax=ax
        )
    else:
        sns.boxplot(data=temp, x=x_axis, y=y_axis, ax=ax)


def plot_grid(grouped_stats, x_axis, y_axis, factors, plotting_function=None, **kwargs):
    """Plot a grid of plots using the specified plotting function.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing the data to plot.
    x_axis : str
        The name of the column to be used for the x-axis.
    y_axis : str
        The name of the column to be used for the y-axis.
    factors : list
        A list of column names to be used as additional factors for grouping.
    plotting_function : callable, optional
        A function to use for plotting, by default None
    height : float, optional
        Height of each facet in inches, by default 1.3
    aspect : float, optional
        Aspect ratio of each facet, by default 1.3
    group_variables : bool, optional
        Whether to aggregate results by computing mean and standard error
        for each combination of factors, by default False
    se_bands : str, optional
        The name of the column to use for plotting standard error bands, by default None
    log_y_axis : bool, optional
        Whether to use a logarithmic scale for the y-axis, by default False
    log_x_axis : bool, optional
        Whether to use a logarithmic scale for the x-axis, by default False
    name_conversion : dict, optional
        A dictionary mapping variable names to more descriptive names for
        axis labels and titles, by default {}
    add_legend : bool, optional
        Whether to add a legend to the plot, by default True
    save_path : str, optional
        Path to save the plot, by default None


    Returns
    -------
    sns.FacetGrid
        The FacetGrid object containing the plots.
    """
    if plotting_function is None:
        raise ValueError("plotting_function must be provided.")

    height = kwargs.get("height", 1.3)
    save_path = kwargs.get("save_path", None)
    se_bands = kwargs.get("se_bands", None)
    log_y_axis = kwargs.get("log_y_axis", False)
    log_x_axis = kwargs.get("log_x_axis", False)
    aspect = kwargs.get("aspect", 1.3)
    name_conversion = kwargs.get("name_conversion", {})
    add_legend = kwargs.get("add_legend", True)
    title = kwargs.get("title", None)
    x_axis_title = kwargs.get("x_axis_title", None)
    y_axis_title = kwargs.get("y_axis_title", None)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    hue_variable = factors[0] if len(factors) >= 2 else None
    aggregate_x = factors[1] if len(factors) >= 2 else factors[0]
    aggregate_y = factors[2] if len(factors) >= 3 else None

    g = sns.FacetGrid(
        grouped_stats,
        row=aggregate_y,
        col=aggregate_x,
        margin_titles=True,
        sharey=True,
        sharex=True,
        height=height,
        aspect=aspect,
    )

    g.map_dataframe(
        plotting_function,
        x_axis=x_axis,
        y_axis=y_axis,
        factors=factors,
        **kwargs,
    )
    # remove default x/y axis labels and tick labels from all subplots
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    # Set x and y axis labels only in central places
    if x_axis_title is None:
        x_axis_title = (
            "Log " + name_conversion.get(x_axis, x_axis).replace("_", " ").title()
            if log_x_axis
            else name_conversion.get(x_axis, x_axis).replace("_", " ").title()
        )
    g.axes[-1, g.axes.shape[1] // 2].set_xlabel(x_axis_title)

    if y_axis_title is None:
        y_axis_title = (
            "Log " + name_conversion.get(y_axis, y_axis).replace("_", " ").title()
            if log_y_axis
            else name_conversion.get(y_axis, y_axis).replace("_", " ").title()
        )

    g.axes[g.axes.shape[0] // 2, 0].set_ylabel(y_axis_title)

    if len(factors) >= 2:
        # column facet titles
        for ax in range(g.axes.shape[1]):
            # put percentage sign for fraction variables

            if re.search(
                r"(?<![a-z])(?:percentage|fraction|prop)(?![a-z])",
                aggregate_x,
                re.IGNORECASE,
            ):
                facet_title = f"{int(g.col_names[ax] * 100)}% {name_conversion.get(aggregate_x, aggregate_x).replace('_', ' ').title()}"
            elif aggregate_x[0] == "_":
                facet_title = ""
            else:
                facet_title = f"{name_conversion.get(aggregate_x, aggregate_x).replace('_', ' ').title()}: {g.col_names[ax]}"
            g.axes[0, ax].set_title(facet_title)

        # custom row facet labels
        if aggregate_y is not None:
            for ax in range(g.axes.shape[0]):
                # put percentage sign for fraction variables
                if re.search(
                    r"(?<![a-z])(?:percentage|fraction|prop)(?![a-z])",
                    aggregate_y,
                    re.IGNORECASE,
                ):
                    text = f"{int(g.row_names[ax] * 100)}\\% {name_conversion.get(aggregate_y, aggregate_y).replace('_', ' ').title()}"
                else:
                    text = f"{name_conversion.get(aggregate_y, aggregate_y).replace('_', ' ').title()}: {g.row_names[ax]}"
                g.axes[ax, -1].texts[0].set_text(text)

    if add_legend is True:
        g.add_legend()

    # code to make the title centered above the grid not the legend
    plot_center_x = (
        g.axes[0, 0].get_position().x0 + g.axes[0, -1].get_position().x1
    ) / 2
    if title is not None:
        g.figure.suptitle(
            title,
            y=1.02,
            x=plot_center_x,
        )
    else:
        if log_y_axis is True:
            g.figure.suptitle(
                "Log "
                + name_conversion.get(x_axis, x_axis)
                + " vs Log "
                + name_conversion.get(y_axis, y_axis)
                if log_x_axis
                else name_conversion.get(x_axis, x_axis)
                + " vs Log "
                + name_conversion.get(y_axis, y_axis),
                y=1.02,
                x=plot_center_x,
            )
        else:
            g.figure.suptitle(
                "Log "
                + name_conversion.get(x_axis, x_axis)
                + " vs "
                + name_conversion.get(y_axis, y_axis)
                if log_x_axis
                else name_conversion.get(x_axis, x_axis)
                + " vs "
                + name_conversion.get(y_axis, y_axis),
                y=1.02,
                x=plot_center_x,
            )

    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return g
