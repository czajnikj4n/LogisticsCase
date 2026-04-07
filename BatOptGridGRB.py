import mip
import pandas as pd
import matplotlib.pyplot as plt
import time
import gurobipy


class Solver:

    def __init__(self) -> None:
        print("Initializing Solver")

        self.df = pd.read_csv(
            "/Users/janekczajnik/Desktop/Erasmus/B3/Introductory Seminar CS/net_demand_and_price.csv"
            # replace with your data file path
        )

        self.parameters = {
            "initial_charge": 2,
            "min_charge": 0.5,
            "max_charge": 5,
            "charging_power_limit": 2,
            "discharging_power_limit": 2,
            "charging_efficiency": 0.95,
            "discharging_efficiency": 0.95
        }

        self.grid_fee = 0
        self.solver_name = mip.GUROBI

    def solve_bo_mip(self):
        price = self.df["Price (EUR/kWh)"].to_numpy()
        demand = self.df["Volume (kWh)"].to_numpy()

        T = len(self.df)
        Delta = 1
        S0 = self.parameters["initial_charge"]
        S_min = self.parameters["min_charge"]
        S_max = self.parameters["max_charge"]
        P_C = self.parameters["charging_power_limit"]
        P_D = self.parameters["discharging_power_limit"]
        eta_c = self.parameters["charging_efficiency"]
        eta_d = self.parameters["discharging_efficiency"]
        grid_fee = self.grid_fee

        bo_mip = mip.Model("BO-MIP", sense=mip.MINIMIZE, solver_name=self.solver_name)
        bo_mip.verbose = 0
        bo_mip.solver.set_int_param("OutputFlag", 0)

        # Variables: formulation-style indexing
        # e[0] is initial SOC, and for t=1..T:
        # e[t] = SOC after period t
        e = {
            t: bo_mip.add_var(lb=0, name=f"e_{t}")
            for t in range(T + 1)
        }
        p_c, p_d, delta = {}, {}, {}
        for t in range(1, T + 1):
            p_c[t] = bo_mip.add_var(lb=0, ub=P_C, name=f"p_c_{t}")
            p_d[t] = bo_mip.add_var(lb=0, ub=P_D, name=f"p_d_{t}")
            delta[t] = bo_mip.add_var(var_type=mip.BINARY, name=f"delta_{t}")


        # Objective
        bo_mip.objective = mip.xsum(
            price[t - 1] * demand[t - 1]
            + ((price[t - 1] + grid_fee) * p_c[t] * Delta)
            - ((price[t - 1] - grid_fee) * p_d[t] * Delta)
            for t in range(1, T + 1)
        )

        # Constraints
        bo_mip += e[0] == S0

        for t in range(1, T + 1):
            bo_mip += (
                    e[t] == e[t - 1]
                    + eta_c * p_c[t] * Delta
                    - (1 / eta_d) * p_d[t] * Delta
            )

            # Basic formulation: only SOC bounds on the state
            bo_mip += e[t] >= S_min
            bo_mip += e[t] <= S_max

            bo_mip += p_c[t] <= P_C * delta[t]
            bo_mip += p_d[t] <= P_D * (1 - delta[t])

        start_time = time.perf_counter()
        bo_mip.optimize()
        solve_time = time.perf_counter() - start_time


        # Extract solved values
        charge_vals = [p_c[t].x * Delta for t in range(1, T + 1)]
        discharge_vals = [p_d[t].x * Delta for t in range(1, T + 1)]
        soc_vals = [e[t].x for t in range(T + 1)]

        total_charge_kwh = sum(charge_vals)
        total_discharge_kwh = sum(discharge_vals)

        usable_capacity = S_max - S_min
        equivalent_cycles = (
            total_discharge_kwh / usable_capacity if usable_capacity > 0 else 0.0
        )

        eps = 1e-6
        charge_periods = sum(1 for x in charge_vals if x > eps)
        discharge_periods = sum(1 for x in discharge_vals if x > eps)
        idle_periods = T - charge_periods - discharge_periods

        bo_mip_results_summary = pd.DataFrame([{
            "model": "BO-MIP",   # change to "BO-MIP" in the BO method
            "grid_fee": grid_fee,
            "solver_objective": bo_mip.objective_value,   # change to bo_mip.objective_value in BO
            "solve_time_sec": solve_time,
            "total_charge_kwh": total_charge_kwh,
            "total_discharge_kwh": total_discharge_kwh,
            "equivalent_cycles": equivalent_cycles,
            "avg_soc": sum(soc_vals) / len(soc_vals),
            "soc_std": pd.Series(soc_vals).std(),
            "charge_periods": charge_periods,
            "discharge_periods": discharge_periods,
            "idle_periods": idle_periods,
            "final_soc": e[T].x,
        }])

        soc_vals = [e[t].x for t in range(T + 1)]
        detail_df = pd.DataFrame({
            "Start": pd.to_datetime(self.df["Start"]),
            "End": pd.to_datetime(self.df["End"]),
            "price": price,
            "demand": demand,
            "charge": charge_vals,
            "discharge": discharge_vals,
            "soc_before": soc_vals[:-1],
            "soc_after": soc_vals[1:],
        })
        detail_df["delta_soc"] = detail_df["soc_after"] - detail_df["soc_before"]
        detail_df["model"] = "BO-MIP"   # use "TO-MIP" in solve_to_mip
        detail_df["grid_fee"] = grid_fee

        self.last_detail_results = detail_df

        # print(bo_mip_results_summary.to_string(index=False))
        return bo_mip_results_summary

    def solve_to_mip(self):
        price = self.df["Price (EUR/kWh)"].to_numpy()
        demand = self.df["Volume (kWh)"].to_numpy()

        T = len(self.df)
        Delta = 1
        S0 = self.parameters["initial_charge"]
        S_min = self.parameters["min_charge"]
        S_max = self.parameters["max_charge"]
        P_C = self.parameters["charging_power_limit"]
        P_D = self.parameters["discharging_power_limit"]
        eta_c = self.parameters["charging_efficiency"]
        eta_d = self.parameters["discharging_efficiency"]

        grid_fee = self.grid_fee

        to_mip = mip.Model("TO-MIP", sense=mip.MINIMIZE, solver_name=self.solver_name)
        to_mip.verbose = 0
        to_mip.solver.set_int_param("OutputFlag", 0)

        # Variables: same indexing as BO-MIP
        e = {
            t: to_mip.add_var(lb=0, name=f"e_{t}")
            for t in range(T + 1)
        }
        p_c, p_d, delta = {}, {}, {}
        for t in range(1, T + 1):
            p_c[t] = to_mip.add_var(lb=0, ub=P_C, name=f"p_c_{t}")
            p_d[t] = to_mip.add_var(lb=0, ub=P_D, name=f"p_d_{t}")
            delta[t] = to_mip.add_var(var_type=mip.BINARY, name=f"delta_{t}")

        # Objective
        to_mip.objective = mip.xsum(
            price[t - 1] * demand[t - 1]
            + ((price[t - 1] + grid_fee) * p_c[t] * Delta)
            - ((price[t - 1] - grid_fee) * p_d[t] * Delta)
            for t in range(1, T + 1)
        )


        # Constraints
        to_mip += e[0] == S0

        for t in range(1, T + 1):
            to_mip += (
                    e[t] == e[t - 1]
                    + eta_c * p_c[t] * Delta
                    - (1 / eta_d) * p_d[t] * Delta
            )

            # Tight formulation: pre-action feasibility constraints
            to_mip += e[t - 1] - (1 / eta_d) * p_d[t] * Delta >= S_min
            to_mip += e[t - 1] + eta_c * p_c[t] * Delta <= S_max

            to_mip += p_c[t] <= P_C * delta[t]
            to_mip += p_d[t] <= P_D * (1 - delta[t])

        start_time = time.perf_counter()
        to_mip.optimize()
        solve_time = time.perf_counter() - start_time


        # Extract solved values
        charge_vals = [p_c[t].x * Delta for t in range(1, T + 1)]
        discharge_vals = [p_d[t].x * Delta for t in range(1, T + 1)]
        soc_vals = [e[t].x for t in range(T + 1)]

        total_charge_kwh = sum(charge_vals)
        total_discharge_kwh = sum(discharge_vals)

        usable_capacity = S_max - S_min
        equivalent_cycles = (
            total_discharge_kwh / usable_capacity if usable_capacity > 0 else 0.0
        )

        eps = 1e-6
        charge_periods = sum(1 for x in charge_vals if x > eps)
        discharge_periods = sum(1 for x in discharge_vals if x > eps)
        idle_periods = T - charge_periods - discharge_periods

        to_mip_results_summary = pd.DataFrame([{
            "model": "TO-MIP",   # change to "BO-MIP" in the BO method
            "grid_fee": grid_fee,
            "solver_objective": to_mip.objective_value,   # change to bo_mip.objective_value in BO
            "solve_time_sec": solve_time,
            "total_charge_kwh": total_charge_kwh,
            "total_discharge_kwh": total_discharge_kwh,
            "equivalent_cycles": equivalent_cycles,
            "avg_soc": sum(soc_vals) / len(soc_vals),
            "soc_std": pd.Series(soc_vals).std(),
            "charge_periods": charge_periods,
            "discharge_periods": discharge_periods,
            "idle_periods": idle_periods,
            "final_soc": e[T].x,
        }])

        soc_vals = [e[t].x for t in range(T + 1)]
        detail_df = pd.DataFrame({
            "Start": pd.to_datetime(self.df["Start"]),
            "End": pd.to_datetime(self.df["End"]),
            "price": price,
            "demand": demand,
            "charge": charge_vals,
            "discharge": discharge_vals,
            "soc_before": soc_vals[:-1],
            "soc_after": soc_vals[1:],
        })
        detail_df["delta_soc"] = detail_df["soc_after"] - detail_df["soc_before"]
        detail_df["model"] = "BO-MIP"   # use "TO-MIP" in solve_to_mip
        detail_df["grid_fee"] = grid_fee

        self.last_detail_results = detail_df

        # print(to_mip_results_summary.to_string(index=False))
        return to_mip_results_summary


    def plot_statistics(self, results):
        stats = results.copy()

        # Sort for clean plotting
        stats = stats.sort_values(["model", "grid_fee"]).reset_index(drop=True)

        models = stats["model"].unique()

        # 1. Solver objective vs grid fee
        plt.figure(figsize=(10, 4))
        for model in models:
            sub = stats[stats["model"] == model]
            plt.plot(sub["grid_fee"], sub["solver_objective"], marker="o", label=model)
        plt.title("Solver objective vs grid fee")
        plt.xlabel("Grid fee (EUR/kWh)")
        plt.ylabel("Objective value")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2. Solve time vs grid fee
        plt.figure(figsize=(10, 4))
        for model in models:
            sub = stats[stats["model"] == model]
            plt.plot(sub["grid_fee"], sub["solve_time_sec"], marker="o", label=model)
        plt.title("Solve time vs grid fee")
        plt.xlabel("Grid fee (EUR/kWh)")
        plt.ylabel("Seconds")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 3. Total charge and discharge vs grid fee
        plt.figure(figsize=(10, 4))
        for model in models:
            sub = stats[stats["model"] == model]
            plt.plot(sub["grid_fee"], sub["total_charge_kwh"], marker="o", label=f"{model} charge")
            plt.plot(sub["grid_fee"], sub["total_discharge_kwh"], marker="x", label=f"{model} discharge")
        plt.title("Battery throughput vs grid fee")
        plt.xlabel("Grid fee (EUR/kWh)")
        plt.ylabel("kWh")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 4. Equivalent cycles vs grid fee
        plt.figure(figsize=(10, 4))
        for model in models:
            sub = stats[stats["model"] == model]
            plt.plot(sub["grid_fee"], sub["equivalent_cycles"], marker="o", label=model)
        plt.title("Equivalent cycles vs grid fee")
        plt.xlabel("Grid fee (EUR/kWh)")
        plt.ylabel("Equivalent cycles")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 5. Average SOC vs grid fee
        plt.figure(figsize=(10, 4))
        for model in models:
            sub = stats[stats["model"] == model]
            plt.plot(sub["grid_fee"], sub["avg_soc"], marker="o", label=model)
        plt.title("Average SOC vs grid fee")
        plt.xlabel("Grid fee (EUR/kWh)")
        plt.ylabel("Average SOC (kWh)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 6. SOC standard deviation vs grid fee
        plt.figure(figsize=(10, 4))
        for model in models:
            sub = stats[stats["model"] == model]
            plt.plot(sub["grid_fee"], sub["soc_std"], marker="o", label=model)
        plt.title("SOC standard deviation vs grid fee")
        plt.xlabel("Grid fee (EUR/kWh)")
        plt.ylabel("SOC std (kWh)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 7. Active / idle periods vs grid fee
        plt.figure(figsize=(10, 4))
        for model in models:
            sub = stats[stats["model"] == model]
            plt.plot(sub["grid_fee"], sub["charge_periods"], marker="o", label=f"{model} charge periods")
            plt.plot(sub["grid_fee"], sub["discharge_periods"], marker="x", label=f"{model} discharge periods")
            plt.plot(sub["grid_fee"], sub["idle_periods"], marker="s", label=f"{model} idle periods")
        plt.title("Battery activity vs grid fee")
        plt.xlabel("Grid fee (EUR/kWh)")
        plt.ylabel("Number of periods")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 8. Final SOC vs grid fee
        plt.figure(figsize=(10, 4))
        for model in models:
            sub = stats[stats["model"] == model]
            plt.plot(sub["grid_fee"], sub["final_soc"], marker="o", label=model)
        plt.title("Final SOC vs grid fee")
        plt.xlabel("Grid fee (EUR/kWh)")
        plt.ylabel("Final SOC (kWh)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print useful comparison table
        print("\nSummary statistics by model:")
        print(stats.to_string(index=False))

        return stats

    def plot_delta_soc_histogram(self, detail_results):
        stats = detail_results.copy()

        plt.figure(figsize=(8, 4))
        plt.hist(stats["delta_soc"], bins=50)
        plt.title(f"Histogram of ΔSOC = e[t] - e[t-1] ({stats['model'].iloc[0]}, fee={stats['grid_fee'].iloc[0]:.2f})")
        plt.xlabel("ΔSOC (kWh)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

        Delta = 1
        eta_c = self.parameters["charging_efficiency"]
        eta_d = self.parameters["discharging_efficiency"]
        P_C = self.parameters["charging_power_limit"]
        P_D = self.parameters["discharging_power_limit"]

        theoretical_upper_bound = eta_c * P_C * Delta
        theoretical_lower_bound = -(1 / eta_d) * P_D * Delta

        print("ΔSOC summary:")
        print(stats["delta_soc"].describe())
        print("Theoretical upper bound on ΔSOC:", theoretical_upper_bound)
        print("Theoretical lower bound on ΔSOC:", theoretical_lower_bound)



if __name__ == "__main__":
    solver = Solver()
    fees = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    all_results = []

    for f in fees:
        solver.grid_fee = f
        all_results.append(solver.solve_bo_mip())
        all_results.append(solver.solve_to_mip())

    results = pd.concat(all_results, ignore_index=True)
    print(results.to_string(index=False))
    solver.plot_delta_soc_histogram(solver.last_detail_results)





