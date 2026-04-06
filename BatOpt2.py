import mip
import pandas as pd
import matplotlib.pyplot as plt
import time

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

        bo_mip = mip.Model("Basic", sense=mip.MINIMIZE)
        bo_mip.verbose = 0

        # Variables: formulation-style indexing
        # e[0] is initial SOC, and for t=1..T:
        # e[t] = SOC after period t
        e = {
            t: bo_mip.add_var(lb=S_min, ub=S_max, name=f"e_{t}")
            for t in range(T + 1)
        }
        p_c, p_d, delta = {}, {}, {}
        for t in range(1, T + 1):
            p_c[t] = bo_mip.add_var(lb=0, ub=P_C, name=f"p_c_{t}")
            p_d[t] = bo_mip.add_var(lb=0, ub=P_D, name=f"p_d_{t}")
            delta[t] = bo_mip.add_var(var_type=mip.BINARY, name=f"delta_{t}")

        # Objective
        bo_mip.objective = mip.xsum(
            price[t - 1] * (demand[t - 1] + (p_c[t] - p_d[t]) * Delta)
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

        baseline_cost = float((price * demand).sum())
        optimized_cost = float(sum(
            price[t - 1] * (demand[t - 1] + (p_c[t].x - p_d[t].x) * Delta)
            for t in range(1, T + 1)
        ))

        results_bo_mip = pd.DataFrame([{
            "model": "BO-MIP",
            "baseline_cost": baseline_cost,
            "optimized_cost": optimized_cost,
            "battery_value": baseline_cost - optimized_cost,
            "solver_objective": bo_mip.objective_value,
            "solve_time_sec": solve_time,
        }])

        print(results_bo_mip.to_string(index=False))
        return results_bo_mip

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

        to_mip = mip.Model("Tight", sense=mip.MINIMIZE)
        to_mip.verbose = 0

        # Variables: same indexing as BO-MIP
        e = {
            t: to_mip.add_var(lb=S_min, ub=S_max, name=f"e_{t}")
            for t in range(T + 1)
        }
        p_c, p_d, delta = {}, {}, {}
        for t in range(1, T + 1):
            p_c[t] = to_mip.add_var(lb=0, ub=P_C, name=f"p_c_{t}")
            p_d[t] = to_mip.add_var(lb=0, ub=P_D, name=f"p_d_{t}")
            delta[t] = to_mip.add_var(var_type=mip.BINARY, name=f"delta_{t}")

        # Objective
        to_mip.objective = mip.xsum(
            price[t - 1] * (demand[t - 1] + (p_c[t] - p_d[t]) * Delta)
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

        baseline_cost = float((price * demand).sum())
        optimized_cost = float(sum(
            price[t - 1] * (demand[t - 1] + (p_c[t].x - p_d[t].x) * Delta)
            for t in range(1, T + 1)
        ))

        results_to_mip = pd.DataFrame([{
            "model": "TO-MIP",
            "baseline_cost": baseline_cost,
            "optimized_cost": optimized_cost,
            "battery_value": baseline_cost - optimized_cost,
            "solver_objective": to_mip.objective_value,
            "solve_time_sec": solve_time,
        }])

        print(results_to_mip.to_string(index=False))
        return results_to_mip


    def plot_statistics(self, results):
        stats = results.copy()
        stats["Start"] = pd.to_datetime(stats["Start"])
        stats = stats.sort_values("Start")

        # Use Start as time index
        stats = stats.set_index("Start")

        # Weekly aggregation
        weekly = stats.resample("7D").agg({
            "price": "mean",
            "demand": "mean",
            "charge": "sum",
            "discharge": "sum",
            "soc": "mean",
            "battery_value": "sum",
        })

        # Daily spread first, then average to weekly
        daily = stats.resample("1D").agg({
            "price": ["max", "min"]
        })
        daily.columns = ["price_max", "price_min"]
        daily["daily_spread"] = daily["price_max"] - daily["price_min"]

        weekly_spread = daily["daily_spread"].resample("7D").mean()

        # Merge into one aligned dataframe
        analysis = weekly.join(weekly_spread, how="inner")

        # Optional: drop final incomplete week if it is much shorter / distorted
        if len(analysis) > 1:
            analysis = analysis.iloc[:-1]

        analysis["cumulative_battery_value"] = analysis["battery_value"].cumsum()

        # 1. Weekly battery value
        plt.figure(figsize=(12, 4))
        plt.plot(analysis.index, analysis["battery_value"], marker="o")
        plt.title("Weekly battery value")
        plt.xlabel("Week")
        plt.ylabel("EUR")
        plt.axhline(0, linestyle="--")
        plt.tight_layout()
        plt.show()

        # 2. Cumulative battery value
        plt.figure(figsize=(12, 4))
        plt.plot(analysis.index, analysis["cumulative_battery_value"], marker="o")
        plt.title("Cumulative battery value")
        plt.xlabel("Week")
        plt.ylabel("EUR")
        plt.axhline(0, linestyle="--")
        plt.tight_layout()
        plt.show()

        # 3. Weekly charge / discharge totals
        plt.figure(figsize=(12, 4))
        plt.plot(analysis.index, analysis["charge"], label="weekly charge", marker="o")
        plt.plot(analysis.index, analysis["discharge"], label="weekly discharge", marker="o")
        plt.title("Weekly battery throughput")
        plt.xlabel("Week")
        plt.ylabel("kWh per week")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 4. Weekly mean price
        plt.figure(figsize=(12, 4))
        plt.plot(analysis.index, analysis["price"], marker="o")
        plt.title("Weekly average electricity price")
        plt.xlabel("Week")
        plt.ylabel("EUR/kWh")
        plt.tight_layout()
        plt.show()

        # 5. Weekly mean demand
        plt.figure(figsize=(12, 4))
        plt.plot(analysis.index, analysis["demand"], marker="o")
        plt.title("Weekly average demand")
        plt.xlabel("Week")
        plt.ylabel("kWh")
        plt.axhline(0, linestyle="--")
        plt.tight_layout()
        plt.show()

        # 6. Battery value vs daily-spread-based weekly spread
        fig, ax1 = plt.subplots(figsize=(12, 4))

        ax1.plot(analysis.index, analysis["battery_value"], color="blue", marker="o")
        ax1.set_ylabel("Battery value (EUR)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.plot(analysis.index, analysis["daily_spread"], color="orange", marker="o")
        ax2.set_ylabel("Avg daily spread (EUR/kWh)", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        plt.title("Weekly battery value vs average daily price spread")
        fig.tight_layout()
        plt.show()

        # 7. Correlation scatter
        plt.figure(figsize=(6, 4))
        plt.scatter(analysis["daily_spread"], analysis["battery_value"])
        plt.xlabel("Average daily spread (EUR/kWh)")
        plt.ylabel("Weekly battery value (EUR)")
        plt.title("Does spread drive battery value?")
        plt.tight_layout()
        plt.show()

        # 8. One-week detailed operational view (first full week)
        first_week = stats.iloc[:24 * 7]

        plt.figure(figsize=(12, 4))
        plt.plot(first_week.index, first_week["price"], label="price")
        plt.plot(first_week.index, first_week["soc"], label="soc")
        plt.title("One-week detail: price and SOC")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(first_week.index, first_week["charge"], label="charge")
        plt.plot(first_week.index, first_week["discharge"], label="discharge")
        plt.title("One-week detail: charge and discharge")
        plt.xlabel("Time")
        plt.ylabel("kW")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Print a few useful summary stats
        corr = analysis["daily_spread"].corr(analysis["battery_value"])
        print("Correlation between weekly battery value and average daily spread:", corr)

        # Main Insight: Battery makes money by exploiting intra-day price volatility / spread.
        # Strong correlation between avg weekly battery value & weekly average of daily price spreads.

        return analysis




if __name__ == "__main__":
    solver = Solver()

    solution_bo_mip = solver.solve_bo_mip()
    solution_to_mip = solver.solve_to_mip()


