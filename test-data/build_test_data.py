"""Build the test-data files from the raw sources and seeded generators.

Run from the app folder:  python test-data/build_test_data.py
Real files are reshaped only (renamed columns, trimmed to complete periods);
values are never altered. Synthetic files are reproducible (fixed seeds).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
REAL = HERE / "real"
SYN = HERE / "synthetic"

DIWALI = pd.to_datetime(["2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20"])
HOLI = pd.to_datetime(["2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14"])


def fred(series_id: str) -> pd.DataFrame:
    df = pd.read_csv(SRC / f"fred_{series_id}.csv")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna()


def save(df: pd.DataFrame, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xlsx":
        df.to_excel(path, index=False, sheet_name=kw.pop("sheet_name", "Sales"))
    else:
        df.to_csv(path, index=False, **kw)
    print(f"{path.relative_to(HERE)}  ({len(df)} rows)")


# ======================================================================== real

def build_real() -> None:
    bikes = pd.read_csv(SRC / "seattle_fremont_bridge_daily.csv", parse_dates=["date"])
    weather = pd.read_csv(SRC / "open_meteo_seattle_daily.csv", parse_dates=["date"])
    daily = bikes.merge(weather, on="date")
    save(daily.rename(columns={"crossings": "bike_crossings", "temperature_2m_max_c": "max_temp_c",
                               "precipitation_sum_mm": "rain_mm"})
         .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")),
         REAL / "01_daily_seattle_bike_crossings_with_weather.csv")

    # Monday-start weeks; keep complete weeks only (2023-01-02 .. 2024-12-29)
    d = daily[(daily["date"] >= "2023-01-02") & (daily["date"] <= "2024-12-29")]
    weekly = d.set_index("date").resample("W-MON", label="left", closed="left").agg(
        {"crossings": "sum", "temperature_2m_max_c": "mean", "precipitation_sum_mm": "sum"})
    weekly.index = weekly.index.strftime("%Y-%m-%d")
    save(weekly.round(1).reset_index().rename(columns={"date": "week_starting", "crossings": "bike_crossings",
                                                       "temperature_2m_max_c": "avg_max_temp_c",
                                                       "precipitation_sum_mm": "rain_mm"}),
         REAL / "02_weekly_seattle_bike_crossings.csv")

    monthly = {
        "03_monthly_us_ice_cream_production_index.csv": ("IPN31152N", "month", "ice_cream_production_index", "1990"),
        "04_monthly_us_clothing_store_sales_usd_mn.csv": ("MRTSSM448USN", "month", "sales_usd_mn", None),
        "05_monthly_us_light_vehicle_sales_k_units.csv": ("TOTALNSA", "month", "vehicles_sold_thousands", None),
        "06_monthly_us_electricity_generation_index.csv": ("IPG2211A2N", "month", "electricity_output_index", "1990"),
        "07_monthly_us_housing_starts_k_units.csv": ("HOUSTNSA", "month", "housing_starts_thousands", "1990"),
        "08_monthly_us_beer_wine_wholesale_usd_mn.csv": ("S4248SM144NCEN", "month", "wholesale_sales_usd_mn", None),
    }
    for name, (sid, date_col, value_col, since) in monthly.items():
        df = fred(sid)
        if since:
            df = df[df["date"] >= since]
        save(pd.DataFrame({date_col: df["date"].dt.strftime("%Y-%m"), value_col: df["value"]}), REAL / name)

    q = fred("ECOMNSA")
    save(pd.DataFrame({"quarter": q["date"].dt.to_period("Q").astype(str), "ecommerce_sales_usd_mn": q["value"]}),
         REAL / "09_quarterly_us_ecommerce_sales_usd_mn.csv")

    v = fred("TOTALNSA")
    counts = v.groupby(v["date"].dt.year)["value"].count()
    full_years = counts[counts == 12].index
    yearly = v[v["date"].dt.year.isin(full_years)].groupby(v["date"].dt.year)["value"].sum().round(1)
    save(pd.DataFrame({"year": yearly.index, "vehicles_sold_thousands": yearly.values}),
         REAL / "10_yearly_us_light_vehicle_sales_k_units.csv")

    parts = []
    for sku, sid in (("CLOTHING", "MRTSSM448USN"), ("BEER-WINE-WHOLESALE", "S4248SM144NCEN")):
        df = fred(sid)
        df = df[df["date"] >= "2010-01-01"]
        parts.append(pd.DataFrame({"SKU": sku, "Month": df["date"].dt.strftime("%b-%Y"),
                                   "Sales (USD mn)": df["value"]}))
    save(pd.concat(parts, ignore_index=True), REAL / "11_monthly_us_retail_two_categories_SKU.csv")


# =================================================================== synthetic

def festive_lift(dates: pd.DatetimeIndex, events: pd.DatetimeIndex, before: int, after: int, size: float):
    lift = np.zeros(len(dates))
    for e in events:
        days = (dates - e).days
        window = (days >= -before) & (days <= after)
        # ramps up to the festival, drops quickly after
        lift[window] += size * np.where(days[window] <= 0, 1 + days[window] / (before + 1), 0.4)
    return lift


def fmcg_weekly_multi_sku() -> pd.DataFrame:
    """Modern-trade FMCG sell-out, weekly, four SKUs, Excel. Price and promotion drivers."""
    rng = np.random.default_rng(2021)
    weeks = pd.date_range("2022-01-03", "2025-06-30", freq="W-MON")
    t = np.arange(len(weeks))
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    skus = {
        # name: base units/week, growth/yr, price, elasticity, season amplitude, season peak week, promo lift
        "TEA-PREMIUM-250G": (5200, 0.07, 145.0, -1.1, 0.10, 2, 0.28),
        "SHAMPOO-ANTIDANDRUFF-180ML": (3100, 0.05, 199.0, -1.6, 0.12, 30, 0.35),
        "DETERGENT-MATIC-1KG": (4100, 0.09, 230.0, -1.3, 0.06, 44, 0.30),
        "BISCUIT-CREAM-75G": (12500, 0.03, 30.0, -0.8, 0.08, 44, 0.20),
    }
    rows = []
    for sku, (base, growth, price0, elast, amp, peak, lift) in skus.items():
        price = np.full(len(weeks), price0)
        for i, d in enumerate(weeks):  # list-price increases each April (input-cost inflation)
            price[i] = price0 * (1.06 ** (d.year - 2022 + (d.month >= 4) - 1)) if d.year > 2022 or d.month >= 4 \
                else price0
        promo = np.zeros(len(weeks), dtype=int)
        starts = rng.choice(np.arange(2, len(weeks) - 2), size=len(weeks) // 9, replace=False)
        for s in starts:
            promo[s: s + rng.integers(1, 3)] = 1  # 1-2 week promotions
        promo_price = np.where(promo == 1, price * rng.uniform(0.82, 0.9, len(weeks)), price)
        post_promo = np.r_[0, promo[:-1]] * (1 - promo)  # pantry-loading dip the week after
        season = 1 + amp * np.cos(2 * np.pi * (woy - peak) / 52)
        festive = 1 + festive_lift(weeks, DIWALI, 21, 7, 0.25 if "DETERGENT" in sku or "BISCUIT" in sku else 0.1)
        level = base * (1 + growth) ** (t / 52)
        units = (level * season * festive * (promo_price / price0) ** elast * (1 + lift * promo)
                 * (1 - 0.15 * post_promo) * rng.lognormal(0, 0.06, len(weeks)))
        units = np.round(units)
        if "SHAMPOO" in sku:  # distributor stock-out: three weeks of zero sell-out
            units[95:98] = 0
        frame = pd.DataFrame({"Week": weeks.strftime("%d-%m-%Y"), "SKU": sku, "Units": units.astype(int),
                              "Shelf price (INR)": np.round(promo_price, 2), "On promotion": promo})
        if "BISCUIT" in sku:  # new variant launched later: short history
            frame = frame[weeks >= "2024-01-01"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def pharmacy_daily_messy() -> str:
    """Single pharmacy store, cough-and-cold syrup, daily POS export: semicolon CSV,
    DD/MM/YYYY dates, closed on some festival days, a few rows uploaded twice."""
    rng = np.random.default_rng(73)
    days = pd.date_range("2023-04-01", "2025-03-31", freq="D")
    doy = days.day_of_year.to_numpy()
    winter = 1 + 0.55 * np.exp(-((((doy - 15) + 182) % 365 - 182) / 40.0) ** 2)  # Dec-Feb peak
    monsoon = 1 + 0.25 * np.exp(-((doy - 220) / 30.0) ** 2)  # Jul-Sep flu wave
    weekday = np.array([1.05, 1.0, 1.0, 1.0, 1.02, 0.95, 0.7])[days.day_of_week]
    level = 38 * (1 + 0.08 * np.arange(len(days)) / 365)
    units = rng.poisson(level * winter * monsoon * weekday)
    df = pd.DataFrame({"Bill Date": days.strftime("%d/%m/%Y"), "Item": "COUGH SYRUP 100ML",
                       "Qty Sold": units, "MRP": np.where(days < "2024-04-01", 118.0, 126.0)})
    closed = set(pd.to_datetime(["2023-11-12", "2023-11-13", "2024-03-25", "2024-11-01", "2024-11-02"]))
    df = df[~days.isin(closed)]
    dup = df.sample(4, random_state=5)
    dup = dup.assign(**{"Qty Sold": dup["Qty Sold"] + rng.integers(-3, 4, len(dup))})
    df = pd.concat([df, dup]).sort_index(kind="stable")
    big = df["Qty Sold"].astype(int).map(lambda v: f"{v:,}")
    return df.assign(**{"Qty Sold": big}).to_csv(index=False, sep=";")


def spare_parts_intermittent() -> pd.DataFrame:
    """Slow-moving spare part at a regional warehouse: many zero months."""
    rng = np.random.default_rng(11)
    months = pd.period_range("2019-01", "2025-06", freq="M")
    occurs = rng.random(len(months)) < 0.32
    sizes = rng.geometric(0.35, len(months)) * rng.choice([1, 1, 1, 2, 4], len(months))
    return pd.DataFrame({"month": months.astype(str), "part_no": "HYD-PUMP-SEAL-KIT-07",
                         "units_issued": np.where(occurs, sizes, 0)})


def electronics_with_returns() -> pd.DataFrame:
    """Online electronics seller, net monthly units of a phone accessory: festive sale
    spikes and one month of net negative sales after a recall."""
    rng = np.random.default_rng(404)
    months = pd.period_range("2020-01", "2025-08", freq="M")
    t = np.arange(len(months))
    m = months.month.to_numpy()
    sale = np.isin(m, [10]).astype(float) * 1.9 + np.isin(m, [11]).astype(float) * 0.6 + np.isin(m, [1, 7]) * 0.3
    units = np.round(900 * (1 + 0.025) ** t * (1 + sale) * rng.lognormal(0, 0.09, len(t)))
    covid = months.isin(pd.period_range("2020-04", "2020-05", freq="M"))
    units[covid] = np.round(units[covid] * 0.12)
    recall = list(months.astype(str)).index("2023-02")
    units[recall] = -340  # returns exceeded sales
    return pd.DataFrame({"Month": months.strftime("%Y-%m"), "Net units": units.astype(int)})


def cement_monthly_with_future_driver() -> pd.DataFrame:
    """Regional cement dispatches (thousand tonnes) driven by monsoon rainfall,
    with 6 future months where only the rainfall forecast is known."""
    rng = np.random.default_rng(8)
    months = pd.period_range("2016-04", "2025-09", freq="M")
    m = months.month.to_numpy()
    normal_rain = np.array([18, 22, 30, 25, 45, 160, 290, 260, 170, 75, 25, 12], dtype=float)  # mm, Maharashtra-like
    rain = normal_rain[m - 1] * rng.lognormal(0, 0.25, len(m))
    t = np.arange(len(m))
    fy_end = np.isin(m, [3]).astype(float) * 0.12  # March year-end push
    tonnes = (1450 * (1 + 0.06) ** (t / 12) * (1 + fy_end) * (1 - 0.0011 * rain) * rng.lognormal(0, 0.04, len(m)))
    covid = months.isin(pd.period_range("2020-04", "2020-05", freq="M"))
    tonnes[covid] *= np.array([0.15, 0.55])
    hist = pd.DataFrame({"month": months.strftime("%Y-%m"), "dispatch_kt": np.round(tonnes, 1),
                         "rainfall_mm": np.round(rain, 1)})
    future_months = pd.period_range(months[-1] + 1, periods=6, freq="M")
    future = pd.DataFrame({"month": future_months.strftime("%Y-%m"), "dispatch_kt": np.nan,
                           "rainfall_mm": normal_rain[future_months.month.to_numpy() - 1]})
    return pd.concat([hist, future], ignore_index=True)


def two_wheeler_yearly_short() -> pd.DataFrame:
    """Two-wheeler registrations in one Indian state ('000), yearly, as year numbers."""
    rng = np.random.default_rng(5)
    years = np.arange(2009, 2025)
    growth = np.array([0.18, 0.21, 0.14, 0.09, 0.05, 0.08, 0.06, 0.09, 0.11, 0.07, -0.14, -0.13, -0.02, 0.17, 0.12,
                       0.10])
    level = 820 * np.cumprod(1 + growth + rng.normal(0, 0.01, len(years)))
    return pd.DataFrame({"Year": years, "Registrations ('000)": np.round(level, 1)})


def build_synthetic() -> None:
    save(fmcg_weekly_multi_sku(), SYN / "12_weekly_fmcg_four_skus_price_promo.xlsx", sheet_name="Sell-out")
    path = SYN / "13_daily_pharmacy_pos_export_messy.csv"
    path.write_text(pharmacy_daily_messy(), encoding="utf-8")
    print(f"{path.relative_to(HERE)}")
    save(spare_parts_intermittent(), SYN / "14_monthly_spare_part_intermittent.csv")
    save(electronics_with_returns(), SYN / "15_monthly_electronics_net_units_with_returns.csv")
    save(cement_monthly_with_future_driver(), SYN / "16_monthly_cement_dispatch_rainfall_future_rows.csv")
    save(two_wheeler_yearly_short(), SYN / "17_yearly_two_wheeler_registrations.csv")


if __name__ == "__main__":
    build_real()
    build_synthetic()
