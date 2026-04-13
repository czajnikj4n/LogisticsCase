import mip
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


def load_data():
    demand_path = "/Users/balintkovacs/Documents/GitHub/LogisticsCase/net_demand_and_price.csv"
    forecast_path = "/Users/balintkovacs/Documents/GitHub/LogisticsCase/open-meteo-52.11N5.19E4m.csv"

    real_df = pd.read_csv(demand_path)
    real_df["Start"] = pd.to_datetime(real_df["Start"])

    forecast_raw = pd.read_csv(forecast_path, skiprows=3)
    forecast_raw["time"] = pd.to_datetime(forecast_raw["time"])
    forecast_raw = forecast_raw.set_index("time")

    forecast_df = forecast_raw.reindex(real_df["Start"])
    return real_df, forecast_df


def get_d_base(real_df, t):
    current_time = real_df.iloc[t]["Start"]
    midnight_today = current_time.normalize()
    midnight_yesterday = midnight_today - pd.Timedelta(days=1)

    mask = (real_df["Start"] >= midnight_yesterday) & (real_df["Start"] < midnight_yesterday + pd.Timedelta(hours=4))
    night_rows = real_df.loc[mask, "Volume (kWh)"]

    if len(night_rows) == 0:
        return 0.1
    return night_rows.mean()


def optimize_step(current_soc, prices, radiation, d_base, d_min_constraint, max_capacity=5.0):
    S_min, S_max = 0.5, max_capacity
    P_C, P_D = 2.0, 2.0
    eta_c, eta_d = 0.95, 0.95
    c_mult = 0.002815515
    horizon = len(prices)

    hat_d = []
    for rad in radiation:
        val = 0 if np.isnan(rad) else rad
        hat_d.append(d_base - (val * c_mult))

    model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
    model.verbose = 0

    e = [model.add_var() for _ in range(horizon + 1)]
    p_c = [model.add_var(lb=0, ub=P_C) for _ in range(horizon)]
    p_d = [model.add_var(lb=0, ub=P_D) for _ in range(horizon)]
    delta = [model.add_var(var_type=mip.BINARY) for _ in range(horizon)]
    target_soc = S_min + 0.5 * (S_max - S_min)

    dev_plus = model.add_var(lb=0, name="dev_plus")
    dev_minus = model.add_var(lb=0, name="dev_minus")

    model += e[horizon] - target_soc == dev_plus - dev_minus
    model += e[0] == current_soc

    for i in range(horizon):
        model += e[i + 1] == e[i] + eta_c * p_c[i] - (1 / eta_d) * p_d[i]
        model += e[i] - (1 / eta_d) * p_d[i] >= S_min
        model += e[i] + eta_c * p_c[i] <= S_max
        model += p_c[i] <= P_C * delta[i]
        model += p_d[i] <= P_D * (1 - delta[i])
        if hat_d[i] < d_min_constraint:
            model += p_d[i] == 0

    model.objective = mip.xsum(prices[i] * (hat_d[i] + p_c[i] - p_d[i]) for i in range(horizon))
    model.optimize()

    return p_c[0].x, p_d[0].x, hat_d[0]


def run_simulation(real_df, forecast_df, horizon_setting, d_min_val, max_capacity):
    soc = 2.0
    results_log = []
    T = len(real_df)
    start_wall_time = time.time()

    for t in range(T):
        current_time = real_df.iloc[t]["Start"]
        current_hour = current_time.hour

        if horizon_setting == "before/after14rule":
            if current_hour < 14:
                h = 24 - current_hour
            else:
                h = (24 - current_hour) + 24
        else:
            h = int(horizon_setting)

        if t + h > T:
            h = T - t
        if h <= 0:
            break

        prices = real_df.iloc[t: t + h]["Price (EUR/kWh)"].values
        radiation = forecast_df.iloc[t: t + h]["direct_radiation (W/m²)"].values
        d_base = get_d_base(real_df, t)

        pc_act, pd_act, hat_act = optimize_step(soc, prices, radiation, d_base, d_min_val, max_capacity)

        actual_d = real_df.iloc[t]["Volume (kWh)"]
        actual_price = prices[0]

        soc = soc + (0.95 * pc_act) - (pd_act / 0.95)
        soc = max(0.5, min(5.0, soc))

        net_grid = actual_d + pc_act - pd_act
        hourly_cost = actual_price * net_grid
        baseline_cost = actual_price * actual_d

        results_log.append({
            "Start": current_time,
            "price": actual_price,
            "demand": actual_d,
            "forecasted_demand": hat_act,
            "charge": pc_act,
            "discharge": pd_act,
            "soc": soc,
            "hourly_cost": hourly_cost,
            "baseline_hourly_cost": baseline_cost,
            "battery_value": baseline_cost - hourly_cost
        })

    results = pd.DataFrame(results_log)
    elapsed = time.time() - start_wall_time

    plot_soc_charge_diagnostics(results, title_suffix=f"(h={horizon_setting}, d_min={d_min_val}, max_cap={max_capacity})", filename=f"sensitivity_h{horizon_setting}_dmin{d_min_val}_cap{max_capacity}.png")

    summary_stats = {
        "Horizon": horizon_setting,
        "D_min": d_min_val,
        "Runtime": round(elapsed, 3),
        "Baseline_Cost": round(results["baseline_hourly_cost"].sum(), 3),
        "Optimized_Cost": round(results["hourly_cost"].sum(), 3),
        "Battery_Value": round(results["battery_value"].sum(), 3),
        "Total_Charged": round(results["charge"].sum(), 3),
        "Total_Discharged": round(results["discharge"].sum(), 3),
        "Avg_SOC": round(results["soc"].mean(), 3)
    }
    return summary_stats

def plot_soc_charge_diagnostics(results, title_suffix="", periods=24*14, start_idx=0, filename=None):
    df = results.iloc[start_idx:start_idx + periods].copy()

    x = range(len(df))

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 1. Price
    axes[0].plot(x, df["price"].to_numpy(), linewidth=1)
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"Electricity price {title_suffix}")

    # 2. SOC
    axes[1].plot(x, df["soc"].to_numpy(), linewidth=1)
    axes[1].set_ylabel("SOC")
    axes[1].set_title("State of charge")

    # 3. Charge and discharge
    axes[2].plot(x, df["charge"].to_numpy(), label="Charge", linewidth=1)
    axes[2].plot(x, -df["discharge"].to_numpy(), label="Discharge", linewidth=1)
    axes[2].set_ylabel("kWh")
    axes[2].set_title("Charging and discharging")
    axes[2].legend()

    # 4. Cumulative charge vs discharge
    axes[3].plot(x, df["charge"].cumsum().to_numpy(), label="Cumulative charge", linewidth=1)
    axes[3].plot(x, df["discharge"].cumsum().to_numpy(), label="Cumulative discharge", linewidth=1)
    axes[3].set_ylabel("kWh")
    axes[3].set_xlabel("Hour index")
    axes[3].set_title("Cumulative charge vs discharge")
    axes[3].legend()

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    real_df, forecast_df = load_data()
    # horizons = [4, 12, "before/after14rule", 72]
    horizons = [12] #, 72]
    # d_mins = [-10, -1, -0.1, 0]
    d_mins = [-10, -0.1]
    capacities = [5, 7.5, 10, 13.5, 15, 20, 23.5, 27, 31]
    all_results = []

    for h in horizons:
        for d in d_mins:
            for c in capacities:
                print("running horizon:" + str(h) + ",d_min:" + str(d) + ", max_cap = " + str(c))
                res = run_simulation(real_df, forecast_df, h, d, c)
                all_results.append(res)

    summary_df = pd.DataFrame(all_results)
    print(summary_df)
    summary_df.to_csv("sensitivity_results_by_capacity.csv", index=False)