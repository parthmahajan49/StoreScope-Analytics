# StoreScope — Retail Sales Analytics Studio

A Streamlit dashboard built around the provided `store_sales_data.csv` dataset and organized to demonstrate the six Data Analytics assignment practicals.


## Dataset
- 100,000 rows, 25 columns (original uploaded file)
- Retail sales, product, customer, shipping, outlet, geography, discounts and profit fields
- Data source: user-provided CSV; the app does not replace it with a different dataset

## Assignment coverage
1. Design Data Warehouse — conceptual fact table and customer/product/geography/store/shipping/date dimensions
2. Analytical Data Preprocessing — OLAP aggregation / slice-dice and aggregation options
3. Apply Data Wrangling — exploratory cleaning checks, data preview, filtering and visualization
4. Regression — linear regression to predict Sales, Profit or Quantity; MAE, RMSE and R²
5. Analysis of Associations — categorical co-occurrence rules with support, confidence and lift
6. Classification — Decision Tree classifier on derived Profit Class; confusion matrix and class metrics

## Run locally (Windows)
1. Install Python 3.10+.
2. Extract the ZIP and open the folder in VS Code.
3. Create environment: `python -m venv .venv`
4. Activate: `.venv\\Scripts\\activate`
5. Install: `pip install -r requirements.txt`
6. Run: `streamlit run app.py`

The app opens at http://localhost:8501.


## Optional: use another dataset
Use the CSV uploader in the sidebar. Sales and Profit columns are required for the executive KPIs. Several charts and practicals are enabled when their matching columns exist.


## Important
This is an academic analytics tool, not a production retail BI or financial forecasting system. Regression/classification outputs are educational and should not be treated as guaranteed predictions.
