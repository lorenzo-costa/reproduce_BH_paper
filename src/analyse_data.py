import pandas as pd
import yaml
from src.helper_functions.analyse_functions import analyse_function


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    sim_path = cfg["data_dir"] + "simulated/"
    output_path = cfg["results_dir"] + "raw/"
    sim_results = pd.read_csv(f"{sim_path}/full_simulation_results.csv")
    analyse_scenario = cfg.get("analyse_scenario", {})

    print("Analysing simulation results...")
    for scenario in analyse_scenario:
        scenario_name = scenario["name"]
        x_axis = scenario["x_axis"]
        y_axis = scenario["y_axis"]
        factors = scenario["factors"]
        group_variables = scenario.get("group_variables", False)
        log_y_axis = scenario.get("log_y_axis", False)
        log_x_axis = scenario.get("log_x_axis", False)
        ratio_variable = scenario.get("ratio_variable", None)

        grouped_stats = analyse_function(
            results=sim_results,
            x_axis=x_axis,
            y_axis=y_axis,
            factors=factors,
            ratio_variable=ratio_variable,
            group_variables=group_variables,
            log_x_axis=log_x_axis,
            log_y_axis=log_y_axis,
        )

        grouped_stats.to_csv(f"{output_path}/{scenario_name}_analysed.csv", index=False)
        print(
            f"Saved analysed results for scenario '{scenario_name}' to {output_path}/{scenario_name}_analysed.csv"
        )
