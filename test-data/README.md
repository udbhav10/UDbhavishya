# Test data

Seventeen files for trying the app: real public data (`real/`) and realistic
synthetic business data (`synthetic/`) built to exercise specific app features.
Upload any of them in the **Data** step (*Upload a file*).

Rebuild everything from the raw sources with:

```bash
python test-data/build_test_data.py
```

## Real data (`real/`)

Values are exactly as published; only column names and date formats were changed
and incomplete periods trimmed. Retrieved 11 Nov 2026.

| File | Frequency | What it shows | Source |
|---|---|---|---|
| `01_daily_seattle_bike_crossings_with_weather.csv` | daily, 2023–2024 | Weekday/weekend pattern, summer peak, holiday dips (25 Dec), Jan-2024 cold snap. Two drivers: max temperature and rain | Bike counts: City of Seattle, *Fremont Bridge Bicycle Counter* (data.seattle.gov, dataset 65db-xm6k), hourly counts summed per day. Weather: Open-Meteo Historical Weather API (ERA5 reanalysis) for 47.65 N, 122.35 W |
| `02_weekly_seattle_bike_crossings.csv` | weekly (Mon start), 104 weeks | Yearly cycle but only two years: Winters is correctly unavailable | Weekly sums of file 01 |
| `03_monthly_us_ice_cream_production_index.csv` | monthly, 1990– | Strong summer seasonality, flat long-run level | FRED `IPN31152N`, Federal Reserve industrial production: ice cream and frozen dessert (2017 = 100, not seasonally adjusted) |
| `04_monthly_us_clothing_store_sales_usd_mn.csv` | monthly, 1992– | December peak, growth, **April 2020 COVID collapse** (outlier test) | FRED `MRTSSM448USN`, US Census Monthly Retail Trade: clothing and accessories stores, USD mn, NSA |
| `05_monthly_us_light_vehicle_sales_k_units.csv` | monthly, 1976– | Seasonality plus business cycles (2008–09, 2020) | FRED `TOTALNSA`, BEA total vehicle sales, thousand units, NSA |
| `06_monthly_us_electricity_generation_index.csv` | monthly, 1990– | Two peaks a year (summer cooling, winter heating) | FRED `IPG2211A2N`, Federal Reserve industrial production: electric power generation (2017 = 100, NSA) |
| `07_monthly_us_housing_starts_k_units.csv` | monthly, 1990– | Seasonality with a deep 2006–2011 cycle | FRED `HOUSTNSA`, US Census new privately-owned housing units started, thousands, NSA |
| `08_monthly_us_beer_wine_wholesale_usd_mn.csv` | monthly, 1992– | Summer and December peaks, steady growth | FRED `S4248SM144NCEN`, US Census merchant wholesalers: beer, wine and distilled beverages, USD mn, NSA |
| `09_quarterly_us_ecommerce_sales_usd_mn.csv` | quarterly, 1999– | Q4 holiday peak, strong growth, 2020 step change | FRED `ECOMNSA`, US Census e-commerce retail sales, USD mn, NSA |
| `10_yearly_us_light_vehicle_sales_k_units.csv` | yearly, 1976– | Trend and cycles with no season | Calendar-year sums of `TOTALNSA` (complete years only) |
| `11_monthly_us_retail_two_categories_SKU.csv` | monthly, 2010–, **2 SKUs** | `SKU` column, `Mon-YYYY` dates | Files 04 and 08 combined |

FRED is the Federal Reserve Bank of St. Louis economic database
(https://fred.stlouisfed.org); the underlying statistics are published by the
agencies named above. The raw downloads are kept unchanged in `sources/`.

## Synthetic data (`synthetic/`)

Generated with fixed seeds in `build_test_data.py`. Each file follows realistic
business patterns, anchored to real calendar events (actual Diwali and Holi
dates, the Indian monsoon, March financial-year end, the 2020 lockdown).

| File | Frequency | Built to test |
|---|---|---|
| `12_weekly_fmcg_four_skus_price_promo.xlsx` | weekly, 4 SKUs | **Excel upload**, `SKU` column, `DD-MM-YYYY` dates, two drivers (shelf price, promotion flag). Price elasticity, April list-price rises, promotion lift with a post-promotion dip, Diwali build-up. Shampoo has a 3-week **stock-out (zeros)**; the biscuit variant **launched in 2024** (short history: Winters unavailable) |
| `13_daily_pharmacy_pos_export_messy.csv` | daily | Raw point-of-sale export: **semicolon separator**, `DD/MM/YYYY` dates, store **closed on festival days (gaps)**, four days **uploaded twice (duplicates)**, winter and monsoon flu peaks, Sunday dip, MRP change as driver |
| `14_monthly_spare_part_intermittent.csv` | monthly | **Intermittent demand**: more than half the months are zero, so MAPE is unusable and Winters unavailable |
| `15_monthly_electronics_net_units_with_returns.csv` | monthly | **Negative net sales** in a recall month, October festive-sale spikes, 2020 lockdown collapse |
| `16_monthly_cement_dispatch_rainfall_future_rows.csv` | monthly | Regression driver (monsoon rainfall cuts dispatches), March year-end push, and **6 future rows** with rainfall but no demand |
| `17_yearly_two_wheeler_registrations.csv` | yearly | **Year numbers as dates** (2009, 2010, …), short series (16 years), 2019–2021 slump |

## Suggested checks

- **01**: daily data, the weekly cycle confirmed by the autocorrelation, regression on temperature vs rain.
- **04**: set a replacement under *Split & clean → Advanced* and watch April 2020 get corrected.
- **12**: switch SKUs in the sidebar; map *On promotion* instead of price as the driver.
- **13**: duplicates block the analysis until you choose how to combine them; then fill the gaps.
- **14 and 15**: watch the warnings for zero and negative demand.
- **16**: the Forecast step's regression uses the file's future rainfall values.
