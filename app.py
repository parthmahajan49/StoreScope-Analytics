import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.analytics import prep_data, build_star_schema, olap_group, regression_analysis, classification_analysis, association_rules

st.set_page_config(page_title="StoreScope | Retail Intelligence", page_icon="🛍️", layout="wide")

# Visual identity distinct from the reference dashboard
st.markdown("""<style>ody, [data-testid='stAppViewContainer'] {background:#f5f7fb;}
[data-testid='stSidebar'] {background:#101d33;}
[data-testid='stSidebar'] * {color:#eef4ff !important;}
.block-container {padding-top:1.25rem; padding-bottom:2rem; max-width:1500px;}
.hero {padding:30px 34px;border-radius:24px;background:linear-gradient(115deg,#152a4d,#346d83);color:white;margin-bottom:20px;}
.hero h1 {font-size:2.3rem;margin:0 0 6px 0;color:white;}
.hero p {opacity:.85;margin:0;}
.metric {background:white;border:1px solid #e7edf5;border-radius:17px;padding:18px 20px;box-shadow:0 5px 20px rgba(22,42,74,.04);}
.metric-label {color:#75859c;font-size:.78rem;text-transform:uppercase;letter-spacing:.07em;font-weight:700;}
.metric-value {color:#132744;font-size:1.75rem;font-weight:800;margin-top:6px;}
.metric-sub {color:#8291a5;font-size:.78rem;margin-top:2px;}
.section-note {color:#8291a5;font-size:.88rem;}
div[data-testid='stMetric'] {background:#fff;border:1px solid #e6edf5;padding:14px 16px;border-radius:16px;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_default():
    p = ROOT / "data" / "store_sales_data.csv"
    return pd.read_csv(p)


st.sidebar.markdown("# 🛍️ StoreScope")
st.sidebar.caption("RETAIL SALES INTELLIGENCE")
st.sidebar.markdown("---")
page = st.sidebar.radio("Workspace", ["Executive Overview", "Sales Explorer", "Customer & Product Trends", "Data Wrangling", "Regression Lab", "Association Mining", "Classification Lab", "Data Warehouse & OLAP", "Data Quality"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**DATA ANALYTICS PROJECT**")
st.sidebar.caption("Academic use • Assignment practicals 01–06")

uploaded = st.sidebar.file_uploader("Upload your CSV dataset (optional)", type=["csv"])
raw = pd.read_csv(uploaded) if uploaded else load_default()
if uploaded:
    st.sidebar.success(f"Using uploaded data · {len(raw):,} rows")
if "Sales" not in raw or "Profit" not in raw:
    st.error("Dataset needs at least Sales and Profit columns to run this dashboard."); st.stop()
data = prep_data(raw)

st.markdown("<div class='hero'><div style='font-size:.75rem;letter-spacing:.15em;font-weight:700;margin-bottom:10px'>RETAIL ANALYTICS WORKSPACE</div><h1>StoreScope · Retail Intelligence</h1><p>Explore sales performance, product mix, customer segments, profit drivers and predictive analytics.</p></div>", unsafe_allow_html=True)

# Filters shared by dashboard pages
with st.sidebar.expander("Global filters", expanded=False):
    if "Region" in data:
        regions = st.multiselect("Region", sorted(data["Region"].dropna().unique()), default=sorted(data["Region"].dropna().unique()))
    else: regions=[]
    if "Year" in data:
        years = st.multiselect("Year", sorted(data["Year"].dropna().unique()), default=sorted(data["Year"].dropna().unique()))
    else: years=[]
filtered = data.copy()
if regions and "Region" in filtered: filtered=filtered[filtered["Region"].isin(regions)]
if years and "Year" in filtered: filtered=filtered[filtered["Year"].isin(years)]

# Header metrics
sales=float(filtered["Sales"].sum()); profit=float(filtered["Profit"].sum()); orders=filtered["Order ID"].nunique() if "Order ID" in filtered else len(filtered); qty=float(filtered["Quantity"].sum()) if "Quantity" in filtered else 0

def metric(label, value, subtitle=""):
    st.markdown(f"<div class='metric'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{subtitle}</div></div>", unsafe_allow_html=True)

def money(x): return f"₹{x:,.0f}"

if page in ["Executive Overview", "Sales Explorer", "Customer & Product Trends"]:
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("Total sales", money(sales), f"{len(filtered):,} filtered records")
    with c2: metric("Total profit", money(profit), f"Margin {profit/sales*100 if sales else 0:.1f}%")
    with c3: metric("Orders", f"{orders:,}", "Unique order IDs")
    with c4: metric("Units sold", f"{qty:,.0f}", "Quantity across filtered rows")

if page == "Executive Overview":
    st.subheader("Performance at a glance")
    left,right=st.columns([1.35,1])
    with left:
        if "Order Date" in filtered:
            monthly=filtered.groupby(filtered["Order Date"].dt.to_period("M").astype(str))["Sales"].sum().reset_index(name="Sales")
            monthly.columns=["Month","Sales"]
            st.plotly_chart(px.area(monthly,x="Month",y="Sales",markers=True,title="Sales trend over time",template="plotly_white"),use_container_width=True)
        else: st.info("Order Date not available in uploaded data.")
    with right:
        if "Category of Goods" in filtered:
            cat=filtered.groupby("Category of Goods")["Sales"].sum().reset_index().sort_values("Sales",ascending=False)
            st.plotly_chart(px.pie(cat,names="Category of Goods",values="Sales",hole=.62,title="Sales mix by category",template="plotly_white"),use_container_width=True)
    l,r=st.columns(2)
    with l:
        if "Region" in filtered:
            reg=filtered.groupby("Region",as_index=False)[["Sales","Profit"]].sum().sort_values("Sales",ascending=False)
            st.plotly_chart(px.bar(reg,x="Region",y="Sales",color="Profit",title="Regional sales and profit",template="plotly_white"),use_container_width=True)
    with r:
        if "Sub-Category" in filtered:
            sub=filtered.groupby("Sub-Category",as_index=False)["Profit"].sum().sort_values("Profit")
            st.plotly_chart(px.bar(sub,x="Profit",y="Sub-Category",orientation="h",title="Profit by sub-category",template="plotly_white"),use_container_width=True)
    st.subheader("Quick data preview")
    st.dataframe(filtered.drop(columns=[c for c in ["Customer Name","Last Name"] if c in filtered],errors="ignore").head(12),use_container_width=True)

elif page == "Sales Explorer":
    st.subheader("Explore and slice your sales records")
    dims=[c for c in ["Region","State","Outlet Type","City Type","Segment","Category of Goods","Sub-Category","Ship Mode","Year"] if c in filtered]
    dim=st.selectbox("Analyze sales by",dims)
    metric_name=st.selectbox("Metric",["Sales","Profit","Quantity","Discount"])
    if metric_name not in filtered: st.warning("Metric not available.")
    else:
        grouped=filtered.groupby(dim,as_index=False)[metric_name].sum().sort_values(metric_name,ascending=False)
        st.plotly_chart(px.bar(grouped,x=dim,y=metric_name,color=metric_name,title=f"{metric_name} by {dim}",template="plotly_white"),use_container_width=True)
        st.dataframe(grouped,use_container_width=True)
    st.download_button("Download filtered records CSV",filtered.to_csv(index=False).encode(),"storescope_filtered.csv","text/csv")

elif page == "Customer & Product Trends":
    st.subheader("Customer, outlet and product patterns")
    a,b=st.columns(2)
    if "Segment" in filtered:
        seg=filtered.groupby("Segment",as_index=False)[["Sales","Profit"]].sum()
        a.plotly_chart(px.bar(seg,x="Segment",y=["Sales","Profit"],barmode="group",title="Segment sales & profit",template="plotly_white"),use_container_width=True)
    if "Outlet Type" in filtered:
        outlet=filtered.groupby("Outlet Type",as_index=False)["Sales"].sum()
        b.plotly_chart(px.pie(outlet,names="Outlet Type",values="Sales",hole=.45,title="Sales by outlet type",template="plotly_white"),use_container_width=True)
    if "Sub-Category" in filtered:
        st.plotly_chart(px.treemap(filtered,path=["Category of Goods","Sub-Category"],values="Sales",color="Profit",title="Product hierarchy performance",template="plotly_white"),use_container_width=True)

elif page == "Data Wrangling":
    st.subheader("Practical 03 · Data wrangling and visualization")
    st.markdown("Apply simple cleaning and transformations, then explore the curated records.")
    clean=filtered.copy()
    if st.button("Run data-wrangling checks",type="primary"):
        st.session_state["wrangling_run"]=True
    if st.session_state.get("wrangling_run"):
        st.success("Wrangling checks completed")
        st.write("Rows",len(clean)," · Columns",len(clean.columns)," · Duplicate rows",int(clean.duplicated().sum()))
        st.dataframe(clean.head(25),use_container_width=True)
        if "Profit" in clean and "Sales" in clean: st.plotly_chart(px.scatter(clean.sample(min(3000,len(clean)),random_state=42),x="Sales",y="Profit",color="Category of Goods" if "Category of Goods" in clean else None,title="Sales vs profit (sample)",template="plotly_white"),use_container_width=True)

elif page == "Regression Lab":
    st.subheader("Practical 04 · Regression modelling")
    st.caption("Predict a numeric business measure using sales-record features. This is an academic model, not a business forecast guarantee.")
    target=st.selectbox("Regression target",[c for c in ["Sales","Profit","Quantity"] if c in filtered])
    feature_options=[c for c in ["Quantity","Discount","Year","Outlet Type","City Type","Category of Goods","Region","Segment","Ship Mode","Sub-Category"] if c in filtered and c!=target]
    default=[c for c in ["Quantity","Discount","Category of Goods","Region","Segment"] if c in feature_options]
    feats=st.multiselect("Features",feature_options,default=default)
    if st.button("Train regression model",type="primary"):
        if len(feats)==0: st.warning("Select at least one feature.")
        else:
            with st.spinner("Training model..."):
                model,metrics,preds=regression_analysis(filtered,target,feats)
            st.session_state["reg_result"]=(metrics,preds)
    if "reg_result" in st.session_state:
        metrics,preds=st.session_state["reg_result"]
        m1,m2,m3=st.columns(3)
        m1.metric("MAE",f"{metrics['MAE']:,.2f}");m2.metric("RMSE",f"{metrics['RMSE']:,.2f}");m3.metric("R²",f"{metrics['R²']:.3f}")
        st.plotly_chart(px.scatter(preds,x="Actual",y="Predicted",title="Actual vs predicted (test sample)",template="plotly_white"),use_container_width=True)
        st.dataframe(preds,use_container_width=True)

elif page == "Association Mining":
    st.subheader("Practical 05 · Association analysis")
    st.caption("Find categorical co-occurrence rules from sampled order rows. Support, confidence and lift are shown.")
    col_options=[c for c in ["Category of Goods","Sub-Category","Segment","Region","Outlet Type","City Type","Ship Mode"] if c in filtered]
    cols=st.multiselect("Transaction attributes",col_options,default=[c for c in ["Category of Goods","Sub-Category","Segment","Region"] if c in col_options])
    s,c=st.columns(2); min_sup=s.slider("Minimum support",.01,.30,.04,.01); min_conf=c.slider("Minimum confidence",.10,.90,.30,.05)
    if st.button("Discover association rules",type="primary"):
        if len(cols)<2: st.warning("Choose at least two attributes.")
        else:
            with st.spinner("Mining frequent co-occurrences..."): rules=association_rules(filtered,cols,min_sup,min_conf)
            st.session_state["rules"]=rules
    if "rules" in st.session_state:
        rules=st.session_state["rules"]
        st.metric("Rules found",len(rules))
        if rules.empty: st.info("No rules match the selected thresholds. Lower support or confidence.")
        else:
            st.dataframe(rules,use_container_width=True)
            st.plotly_chart(px.scatter(rules.head(100),x="Confidence",y="Lift",size="Support",hover_data=["Antecedent","Consequent"],title="Association-rule strength",template="plotly_white"),use_container_width=True)

elif page == "Classification Lab":
    st.subheader("Practical 06 · Decision tree classification")
    st.caption("Classify records into profit tiers derived from the dataset. The target is created for academic analysis.")
    work=filtered.copy()
    if "Profit" in work:
        work["Profit Class"]=pd.cut(work["Profit"],bins=[-np.inf,0,work["Profit"].quantile(.5),work["Profit"].quantile(.8),np.inf],labels=["Loss","Low","Medium","High"],duplicates="drop")
    target=st.selectbox("Classification target",["Profit Class"])
    options=[c for c in ["Sales","Quantity","Discount","Year","Outlet Type","City Type","Category of Goods","Region","Segment","Ship Mode","Sub-Category"] if c in work and c!=target]
    defaults=[c for c in ["Sales","Quantity","Discount","Category of Goods","Region","Segment"] if c in options]
    features=st.multiselect("Predictor features",options,default=defaults)
    depth=st.slider("Maximum tree depth",2,12,5)
    if st.button("Train decision tree",type="primary"):
        if len(features)<1: st.warning("Select one or more predictors.")
        else:
            model,accuracy,cm,report,classes=classification_analysis(work,target,features,depth)
            st.session_state["clf_result"]=(accuracy,cm,report,classes)
    if "clf_result" in st.session_state:
        accuracy,cm,report,classes=st.session_state["clf_result"]
        st.metric("Test accuracy",f"{accuracy:.1%}")
        st.plotly_chart(px.imshow(cm,x=classes,y=classes,text_auto=True,labels=dict(x="Predicted",y="Actual",color="Count"),title="Confusion matrix",template="plotly_white"),use_container_width=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3),use_container_width=True)

elif page == "Data Warehouse & OLAP":
    st.subheader("Practical 01 & 02 · Data warehouse design and OLAP")
    st.markdown("**Warehouse grain:** one row per dataset order / sales record. Measures: Sales, Profit, Quantity, Discount. Dimensions include customer, product, geography, store/outlet, shipping and date.")
    fact,dims=build_star_schema(filtered)
    st.metric("Fact rows",f"{len(fact):,}")
    st.markdown("#### Dimensional model")
    for name,table in dims.items():
        with st.expander(f"{name} · {len(table):,} unique rows"):
            st.dataframe(table.head(20),use_container_width=True)
    st.markdown("#### Interactive OLAP slice / dice / roll-up")
    dimension=st.selectbox("Dimension",[c for c in ["Region","State","Segment","Category of Goods","Sub-Category","Outlet Type","Ship Mode","Year","City Type"] if c in filtered])
    measure=st.selectbox("Measure",["Sales","Profit","Quantity","Discount"])
    agg=st.selectbox("Aggregation",["sum","mean","count"])
    result=olap_group(filtered,dimension,measure,agg)
    st.plotly_chart(px.bar(result,x=dimension,y=measure,title=f"OLAP {agg.upper()} of {measure} by {dimension}",template="plotly_white"),use_container_width=True)
    st.dataframe(result,use_container_width=True)
    st.download_button("Download OLAP result",result.to_csv(index=False).encode(),"olap_result.csv","text/csv")

elif page == "Data Quality":
    st.subheader("Data Quality · Profiling and preparation")
    a,b,c=st.columns(3); a.metric("Rows",f"{len(raw):,}"); b.metric("Columns",len(raw.columns)); c.metric("Duplicate rows",f"{int(raw.duplicated().sum()):,}")
    nulls=raw.isna().sum().sort_values(ascending=False)
    st.plotly_chart(px.bar(nulls.reset_index(),x="index",y=0,title="Missing values by field",template="plotly_white"),use_container_width=True)
    st.markdown("#### Field profile")
    profile=pd.DataFrame({"Column":raw.columns,"Data type":raw.dtypes.astype(str).values,"Missing":raw.isna().sum().values,"Unique":raw.nunique().values})
    st.dataframe(profile,use_container_width=True)
    st.markdown("#### Numeric summary")
    st.dataframe(raw.select_dtypes(include=np.number).describe().T,use_container_width=True)
    st.markdown("#### Dataset notes")
    st.info("The bundled dataset contains no missing values in the original file. Date columns are converted during analysis. Identifiers and names are excluded from predictive feature defaults to avoid unnecessary leakage.")

st.markdown("---")
st.caption("StoreScope · Retail Sales Analytics Project | Built for Data Analytics practicals | Dataset: user-provided store sales CSV")
