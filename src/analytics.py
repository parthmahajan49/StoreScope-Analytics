from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Order Date", "Ship Date", "Sales Date", "Date of Birth"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "Profit" in out and "Sales" in out:
        out["Profit Margin %"] = np.where(out["Sales"] != 0, out["Profit"] / out["Sales"] * 100, 0)
    if "Order Date" in out:
        out["Order Month"] = out["Order Date"].dt.month_name()
        out["Order Quarter"] = out["Order Date"].dt.quarter.astype("Int64").astype(str).replace("<NA>", "Unknown")
    if "Ship Date" in out and "Order Date" in out:
        out["Shipping Days"] = (out["Ship Date"] - out["Order Date"]).dt.days
    return out



def build_star_schema(df: pd.DataFrame):
    data = df.copy()
    fact_cols = [c for c in ["Order ID", "Customer ID", "Product ID", "Sales", "Quantity", "Discount", "Profit", "Year", "Order Date", "Ship Date"] if c in data]
    fact = data[fact_cols].copy()
    dimensions = {}
    mappings = {
        "Dim_Customer": ["Customer ID", "Customer Name", "Last Name", "Date of Birth", "Segment"],
        "Dim_Product": ["Product ID", "Product Name", "Sub-Category", "Category of Goods"],
        "Dim_Geography": ["Country", "Region", "State", "Postal Code", "City Type"],
        "Dim_Store": ["Outlet Type"],
        "Dim_Shipping": ["Ship Mode"],
        "Dim_Date": ["Order Date", "Ship Date", "Sales Date", "Year"],
    }
    for name, cols in mappings.items():
        cols = [c for c in cols if c in data.columns]
        dimensions[name] = data[cols].drop_duplicates().reset_index(drop=True)
    return fact, dimensions



def olap_group(df: pd.DataFrame, dimension: str, measure: str, agg: str = "sum"):
    if dimension not in df or measure not in df:
        return pd.DataFrame()
    if agg == "mean": return df.groupby(dimension, dropna=False)[measure].mean().reset_index().sort_values(measure, ascending=False)
    if agg == "count": return df.groupby(dimension, dropna=False)[measure].count().reset_index().sort_values(measure, ascending=False)
    if agg == "sum": return df.groupby(dimension, dropna=False)[measure].sum().reset_index().sort_values(measure, ascending=False)
    return df.groupby(dimension, dropna=False)[measure].agg(agg).reset_index()


def regression_analysis(df: pd.DataFrame, target: str, features: list[str]):
    work = df[features + [target]].copy().dropna(subset=[target])
    X, y = work[features], work[target]
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)], remainder="drop")
    model = Pipeline([("prep", pre), ("reg", LinearRegression())])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42)
    model.fit(Xtr, ytr); pred = model.predict(Xte)
    metrics = {"MAE": mean_absolute_error(yte, pred), "RMSE": mean_squared_error(yte, pred)**.5, "R²": r2_score(yte, pred)}
    return model, metrics, pd.DataFrame({"Actual": yte.values[:500], "Predicted": pred[:500]})


def classification_analysis(df: pd.DataFrame, target: str, features: list[str], depth: int):
    work = df[features + [target]].copy().dropna(subset=[target])
    X, y = work[features], work[target]
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric),
                           ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    model = Pipeline([("prep", pre), ("tree", DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight="balanced"))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    model.fit(Xtr, ytr); pred = model.predict(Xte)
    return model, accuracy_score(yte, pred), confusion_matrix(yte, pred), classification_report(yte, pred, output_dict=True), list(model.named_steps["tree"].classes_)


def association_rules(df: pd.DataFrame, cols: list[str], min_support: float = .03, min_confidence: float = .25, max_rows: int = 10000):
    # Keep the demonstration efficient on the 100k-row dataset. Build simple item sets from selected categorical columns.
    sample = df[cols].dropna().sample(min(max_rows, len(df)), random_state=42)
    transactions = []
    for row in sample.itertuples(index=False, name=None):
        transactions.append(set(f"{col}={val}" for col, val in zip(cols, row)))
    n = len(transactions)
    counts = {}
    pair_counts = {}
    for trans in transactions:
        for item in trans: counts[item] = counts.get(item, 0) + 1
        items = sorted(trans)
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                pair_counts[(items[i], items[j])] = pair_counts.get((items[i], items[j]), 0) + 1
    rules = []
    for (a,b), cnt in pair_counts.items():
        sup = cnt/n
        if sup < min_support: continue
        conf_ab = cnt/counts[a]
        conf_ba = cnt/counts[b]
        lift_ab = conf_ab/(counts[b]/n) if counts[b] else 0
        lift_ba = conf_ba/(counts[a]/n) if counts[a] else 0
        if conf_ab >= min_confidence: rules.append({"Antecedent":a,"Consequent":b,"Support":sup,"Confidence":conf_ab,"Lift":lift_ab})
        if conf_ba >= min_confidence: rules.append({"Antecedent":b,"Consequent":a,"Support":sup,"Confidence":conf_ba,"Lift":lift_ba})
    return pd.DataFrame(rules).sort_values(["Lift", "Confidence"], ascending=False) if rules else pd.DataFrame(columns=["Antecedent","Consequent","Support","Confidence","Lift"])
