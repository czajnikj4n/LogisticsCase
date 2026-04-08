import mip
import pandas as pd
import numpy as np
import time


def load_data():
    demand_path = r"C:\Users\filip\Desktop\net_demand_and_price.csv"
    forecast_path = r"C:\Users\filip\Downloads\open-meteo-52.11N5.19E4m.csv"

    real_df = pd.read_csv(demand_path)
    real_df["Start"] = pd.to_datetime(real_df["Start"], dayfirst=True)

    forecast_raw = pd.read_csv(forecast_path, skiprows=3)
    forecast_raw["time"] = pd.to_datetime(forecast_raw["time"])
    forecast_raw = forecast_raw.set_index("time")

    forecast_df = forecast_raw.reindex(real_df["Start"])
    return real_df, forecast_df


def get_d_base(real_df, t):
    """ Finds average demand of 00:00-04:00 from the previous night """
    current_time = real_df.iloc[t]["Start"]
    midnight_today = current_time.normalize()
    midnight_yesterday = midnight_today - pd.Timedelta(days=1)

    mask = (real_df["Start"] >= midnight_yesterday) & (real_df["Start"] < midnight_yesterday + pd.Timedelta(hours=4))
    night_rows = real_df.loc[mask, "Volume (kWh)"]

    if len(night_rows) == 0:
        return 0.102263566
    return night_rows.mean()


def optimize_step(current_soc, prices, radiation, d_base):
    S_min, S_max = 0.5, 5.0
    P_C, P_D = 2.0, 2.0
    eta_c, eta_d = 0.95, 0.95
    c_mult = 0.002
    horizon = len(prices)

    hat_d = []
    for rad in radiation:
        val = 0 if np.isnan(rad) else rad
        hat_d.append(d_base - (val * c_mult))

    model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
    model.verbose = 0

    e = [model.add_var(lb=S_min, ub=S_max) for _ in range(horizon + 1)]
    p_c = [model.add_var(lb=0, ub=P_C) for _ in range(horizon)]
    p_d = [model.add_var(lb=0, ub=P_D) for _ in range(horizon)]
    delta = [model.add_var(var_type=mip.BINARY) for _ in range(horizon)]

    model += e[0] == current_soc
    for i in range(horizon):
        model += e[i + 1] == e[i] + eta_c * p_c[i] - (1 / eta_d) * p_d[i]
        model += p_c[i] <= P_C * delta[i]
        model += p_d[i] <= P_D * (1 - delta[i])

    model.objective = mip.xsum(prices[i] * (hat_d[i] + p_c[i] - p_d[i]) for i in range(horizon))
    model.optimize()

    return p_c[0].x, p_d[0].x


def run_simulation():
    real_df, forecast_df = load_data()

    soc = 2.0
    results_log = []
    T = len(real_df)

    start_time = time.time()

    for t in range(T):
        h = 4
        if t + h > T:
            h = T - t
        if h <= 0:
            break

        prices = real_df.iloc[t: t + h]["Price (EUR/kWh)"].values
        radiation = forecast_df.iloc[t: t + h]["direct_radiation (W/m²)"].values
        d_base = get_d_base(real_df, t)

        pc_act, pd_act = optimize_step(soc, prices, radiation, d_base)

        actual_d = real_df.iloc[t]["Volume (kWh)"]
        actual_price = prices[0]

        soc = soc + (0.95 * pc_act) - (pd_act / 0.95)
        soc = max(0.5, min(5.0, soc))

        net_grid = actual_d + pc_act - pd_act
        cost = actual_price * net_grid
        baseline = actual_price * actual_d

        results_log.append({
            "Start": real_df.iloc[t]["Start"],
            "price": actual_price,
            "demand": actual_d,
            "charge": pc_act,
            "discharge": pd_act,
            "soc": soc,
            "hourly_cost": cost,
            "baseline_hourly_cost": baseline,
            "battery_value": baseline - cost
        })

    end_time = time.time()
    elapsed = end_time - start_time

    results = pd.DataFrame(results_log)

    print(f"Total Running Time: {elapsed:.2f} seconds")
    print("Baseline cost: ", round(results["baseline_hourly_cost"].sum(), 4), "EUR")
    print("Optimized cost:", round(results["hourly_cost"].sum(), 4), "EUR")
    print("Battery value: ", round(results["battery_value"].sum(), 4), "EUR")
    print("Total charged: ", round(results["charge"].sum(), 2), "kWh")
    print("Total discharged:", round(results["discharge"].sum(), 2), "kWh")
    print("Average SOC:   ", round(results["soc"].mean(), 4), "kWh")

    results.to_csv("rolling_results_4h.csv", index=False)
    return results


if __name__ == "__main__":
    run_simulation()