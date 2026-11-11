# Udbhavishya

*Udbhav* (origin) + *bhavishya* (the future): see what demand comes next.
By Udbhav Dalal.

A guided demand-forecasting web app. Load a sales history (or a demo product), test
forecasting methods honestly on a hidden test period, and get a recommended forecast
with a confidence range.

Built for the *Demand Forecasting* module (slide 83, "Your forecasting app project").

## What it does

Five steps, each with one chart and few controls:

1. **Data**: upload a CSV/Excel file or choose one of five demo products. Dates are
   detected as daily, weekly, monthly, quarterly or yearly. Duplicates, gaps, zeros
   and negative values are flagged. Optionally roll up to a coarser frequency.
2. **Split & clean**: the latest periods become the test period (default: one
   season). Outliers are flagged in training data only (σ = 1.25 × MAD, or the
   percent version for seasonal data) and optionally corrected.
3. **Models**: SES, Holt (optional damping) and multiplicative Winters with α/β/γ/φ
   sliders and an *Optimise* button (training data only), plus regression on one
   causal driver. Start values are computed one way for every method (Hyndman et
   al., 2008), shown on screen and editable.
4. **Compare**: test-period wMAPE, MAD, MAPE, RMSE, bias, tracking signal and MASE
   for every method. The average of the two best is also tested. The recommendation
   is the lowest wMAPE among methods that beat seasonal naive.
5. **Forecast**: the chosen method is refitted on all data and forecasts ahead with
   a 20%, 60% or 80% range. A tracking-signal chart flags bias beyond ±4. The
   forecast can be downloaded as CSV.

Uploaded data stays in the browser session; nothing is written to disk.

## File format

| Column | Required | Notes |
|---|---|---|
| `date` | yes | Any common format: `2024-01-31`, `2024-01`, `Jan-2024`, `31/01/2024`, or a year like `2024` |
| `demand` | yes | Numbers; thousands separators are fine |
| driver | no | One numeric cause of demand, e.g. price, promotion (1/0), temperature |
| `SKU` | no | Product code. Without it, the whole file is one product |

Column names can differ; you map them in the app. Rows after the last demand value
that carry a driver value but no demand are used as future driver values for
regression. A template can be downloaded in the app (Data step → *File format*).

## Run locally

Requires Python 3.12.

```bash
git clone <your-repo-url> forecast-app
cd forecast-app
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Tests

```bash
pytest
```

The suite covers ingestion, every method (hand-checked examples), metrics against
slide 47, ranges against slide 74, the slide-83 audit checklist
(`tests/test_audit.py`), and headless UI runs through every step for each demo.
`tests/test_workbook_parity.py` checks the equations against the course workbook
when `Forecasting_Classroom_Workbook.xlsx` sits in the folder above the app; it
is skipped otherwise.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repository.
2. Sign in at https://share.streamlit.io with GitHub and choose **Create app**.
3. Pick the repository and branch, set the main file to `app.py`, and under
   *Advanced settings* choose Python 3.12.
4. Deploy. Every push to the branch redeploys automatically.

## Project layout

```
app.py            entry point: sidebar, stepper, navigation
ui/               step screens, charts, messages, session state
engine/           all calculations (no Streamlit), unit-tested
tests/            pytest suite
.streamlit/       theme and server settings
```
