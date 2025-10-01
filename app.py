import os
import json
import time
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import streamlit as st
from databricks import sql
import requests
from datetime import date, timedelta
import duckdb
import re


# -----------------------------
# csv fallback
# -----------------------------

OFFLINE_CSV = "data/ecommerce_orders_sample.csv"
_duck_con = None
st.cache_data.clear()

def offline_available() -> bool:
    return os.path.exists(OFFLINE_CSV)

def get_offline_demo_date_bounds():
    # adjust if your CSV has a different range
    return pd.to_datetime("2023-01-01").date(), pd.to_datetime("2024-12-31").date()

def _prep_duck():
    """Create an in-memory DuckDB table from the CSV with correct column names."""
    con = duckdb.connect()
    con.execute("DROP TABLE IF EXISTS ecommerce_orders")
    con.execute(f"""
        CREATE TABLE ecommerce_orders AS
        SELECT
            CAST(order_id     AS VARCHAR) AS order_id,
            CAST(customer_id  AS VARCHAR) AS customer_id,
            CAST(customer_name AS VARCHAR) AS customer_name,
            CAST(region       AS VARCHAR) AS region,
            CAST(order_date   AS DATE)    AS order_date,
            CAST(ship_date    AS DATE)    AS ship_date,
            CAST(product_id   AS VARCHAR) AS product_id,
            CAST(category     AS VARCHAR) AS category,
            CAST(sub_category AS VARCHAR) AS sub_category,
            CAST(product_name AS VARCHAR) AS product_name,
            CAST(quantity     AS BIGINT)  AS quantity,
            CAST(unit_price   AS DOUBLE)  AS unit_price,
            CAST(discount     AS DOUBLE)  AS discount,
            CAST(sales        AS DOUBLE)  AS sales,
            CAST(profit       AS DOUBLE)  AS profit,
            -- optional; present in some CSVs, safe if missing
            TRY_CAST(latitude  AS DOUBLE) AS latitude,
            TRY_CAST(longitude AS DOUBLE) AS longitude
        FROM read_csv_auto('{OFFLINE_CSV}', HEADER=TRUE, IGNORE_ERRORS=TRUE)
    """)
    return con

def _warehouse_id_from_http_path(http_path: str) -> str:
    return http_path.strip("/").split("/")[-1]

def _looks_like_warehouse_stopped(msg: str) -> bool:
    m = msg.upper()
    return any(x in m for x in [
        "WAREHOUSE_SUSPENDED", "ENDPOINT IS IN STOPPED STATE",
        "WAREHOUSE IS STOPPED", "ENDPOINT_NOT_RUNNING",
        "FAILED TO ESTABLISH A NEW CONNECTION"
    ])

def _looks_like_credit_exhausted(msg: str) -> bool:
    m = msg.upper()
    return "COMMUNITY_EDITION_CREDIT_EXHAUSTED" in m or "FREE DAILY LIMIT" in m

def start_warehouse_and_wait(max_wait_s: int = 120, poll_s: int = 5) -> bool:
    host   = st.secrets["DATABRICKS_HOST"]
    token  = st.secrets["DATABRICKS_TOKEN"]
    wid    = _warehouse_id_from_http_path(st.secrets["DATABRICKS_HTTP_PATH"])
    hdrs   = {"Authorization": f"Bearer {token}"}

    r = requests.post(f"https://{host}/api/2.0/sql/warehouses/{wid}/start", headers=hdrs)
    if r.status_code not in (200, 202):
        st.warning(f"Could not start warehouse: {r.text}")
        return False

    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        s = requests.get(f"https://{host}/api/2.0/sql/warehouses/{wid}", headers=hdrs)
        if s.ok and s.json().get("state") == "RUNNING":
            return True
        time.sleep(poll_s)
    st.warning("Warehouse start timed out.")
    return False
    
# -----------------------------
# Page & Styles
# -----------------------------
st.set_page_config(page_title="E-Commerce Analytics (Databricks)", layout="wide")
st.title("🛍️ E-Commerce Analytics — Databricks")

if offline_available():
    st.markdown(
        "<span style='color:deepskyblue; font-weight:bold'>Offline Demo Mode (CSV snapshot)</span>",
        unsafe_allow_html=True
    )

# -----------------------------
# Connection helpers
# -----------------------------
def connect():
    return sql.connect(
        server_hostname=st.secrets["DATABRICKS_HOST"],
        http_path=st.secrets["DATABRICKS_HTTP_PATH"],
        access_token=st.secrets["DATABRICKS_TOKEN"],
    )

@st.cache_data(ttl=300, show_spinner=False)
def run_query(q: str, params: tuple | None = None, force_offline: bool = False) -> pd.DataFrame:
    """Try live Databricks; on stopped/credit-exhausted, fall back to DuckDB CSV."""
    # --- live executor ---
    def _exec_live():
        from databricks import sql
        with sql.connect(
            server_hostname=st.secrets["DATABRICKS_HOST"],
            http_path=st.secrets["DATABRICKS_HTTP_PATH"],
            access_token=st.secrets["DATABRICKS_TOKEN"],
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(q, params) if params else cur.execute(q)
                cols = [c[0] for c in cur.description] if cur.description else []
                rows = cur.fetchall()
        return pd.DataFrame.from_records(rows, columns=cols) if cols else pd.DataFrame()

    # --- offline executor ---
    def _exec_offline(sql_text: str, sql_params: tuple | None):
        global _duck_con
        if _duck_con is None:
            if not offline_available():
                st.error("Offline CSV not found at data/ecommerce_orders_sample.csv.")
                return pd.DataFrame()
            _duck_con = _prep_duck()

        # rewrite FQTN → local table name
        sql_duck = re.sub(r"\bsandbox\.ecommerce_orders\b", "ecommerce_orders", sql_text, flags=re.IGNORECASE)

        # inline ? params (dates/strings/numbers)
        if sql_params:
            safe_vals = []
            for p in sql_params:
                if isinstance(p, (pd.Timestamp,)):
                    safe_vals.append(f"DATE '{p.date()}'")
                elif hasattr(p, "strftime"):  # date/datetime
                    safe_vals.append(f"DATE '{p.strftime('%Y-%m-%d')}'")
                elif isinstance(p, date):
                    safe_vals.append(f"DATE '{p.strftime('%Y-%m-%d')}'")
                elif isinstance(p, str):
                    safe_vals.append("'" + p.replace("'", "''") + "'")
                else:
                    safe_vals.append(str(p))
            for v in safe_vals:
                sql_duck = sql_duck.replace("?", v, 1)

        return _duck_con.execute(sql_duck).df()

    # force offline (checkbox)
    if force_offline:
        return _exec_offline(q, params)

    # live path with one auto-start + retry, then offline fallback
    try:
        return _exec_live()
    except Exception as e:
        msg = str(e)
        if _looks_like_warehouse_stopped(msg):
            st.info("Waking the SQL Warehouse…")
            if start_warehouse_and_wait():
                try:
                    return _exec_live()
                except Exception:
                    pass
        if _looks_like_credit_exhausted(msg) or offline_available():
            return _exec_offline(q, params)

        st.error("Query failed. Check warehouse status/permissions.")
        return pd.DataFrame()
        
# -----------------------------
# Schema constants (adjust if your catalog/schema differ)
# -----------------------------
CATALOG = "sandbox"
TABLE = "ecommerce_orders"   # columns from your screenshot

FQTN = f"{CATALOG}.{TABLE}"

try:
    min_d, max_d, regions, cats, subs = bootstrap_filters()
except Exception:
    min_d, max_d, regions, cats, subs = date(2021, 1, 1), date.today(), [], [], []

# If min_d somehow ends up after max_d, swap them
if min_d > max_d:
    min_d, max_d = max_d, min_d

# if running offline, override to known demo range so filters aren’t empty
if offline_available():
    min_d, max_d = get_offline_demo_date_bounds()

# -----------------------------
# Load basic ranges for filters
# -----------------------------
@st.cache_data(ttl=300)
def bootstrap_filters():
    def safe_cols(df):
        return {c.lower(): c for c in df.columns} if (df is not None and not df.empty) else {}

    # Defaults if the warehouse is sleeping or query fails
    min_d, max_d = date(2021, 1, 1), date.today()
    regions, cats, subs = [], [], []

    try:
        meta = run_query(f"SELECT MIN(order_date) AS min_date, MAX(order_date) AS max_date FROM {FQTN}")
        m = safe_cols(meta)
        if meta is not None and not meta.empty and "min_date" in m and "max_date" in m:
            md = meta[m["min_date"]][0]
            xd = meta[m["max_date"]][0]
            if pd.notna(md): min_d = pd.to_datetime(md).date()
            if pd.notna(xd): max_d = pd.to_datetime(xd).date()

        def distinct(col):
            df = run_query(f"SELECT DISTINCT {col} AS v FROM {FQTN} WHERE {col} IS NOT NULL ORDER BY 1")
            mm = safe_cols(df)
            return df[mm.get("v", col)].dropna().tolist() if df is not None and not df.empty else []

        regions = distinct("region")
        cats    = distinct("category")
        subs    = distinct("sub_category")
    except Exception:
        # swallow errors and keep defaults
        pass

    # if running offline, override to known demo range so filters aren’t empty
    if offline_available():
        min_d, max_d = get_offline_demo_date_bounds()
    
    return min_d, max_d, regions, cats, subs

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("Filters")
    drange = st.date_input(
        "Order date range",
        value=(max(min_d, max_d - timedelta(days=180)), max_d),
        min_value=min_d, max_value=max_d
    )
    region = st.multiselect("Region", regions, default=regions[:2] if regions else [])
    category = st.multiselect("Category", cats, default=cats[:3] if cats else [])
    subcat = st.multiselect("Sub-Category", subs)

    st.divider()
    st.subheader("Query Options")
    limit_rows = st.slider("Row limit (tables)", 100, 10000, 1000, step=100)

    st.divider()
    if st.button("🔌 Start Warehouse"):
        ok = start_warehouse_and_wait()
        if ok:
            st.success("Warehouse is running.")
        else:
            st.error("Could not start warehouse — check permissions or token.")

    force_offline = st.checkbox("Force Offline Demo (CSV)", value=True)


start_date, end_date = (drange if isinstance(drange, tuple) else (min_d, max_d))

# -----------------------------
# Build WHERE clause safely
# -----------------------------
where = ["order_date BETWEEN ? AND ?"]
params: list = [str(start_date), str(end_date)]

def add_in(col: str, values: list[str]):
    if values:
        placeholders = ",".join(["?"] * len(values))
        where.append(f"{col} IN ({placeholders})")
        params.extend(values)

add_in("region", region)
add_in("category", category)
add_in("sub_category", subcat)
where_sql = " AND ".join(where)

# -----------------------------
# Main queries
# -----------------------------
base_sql = f"""
SELECT
  order_id,
  customer_id,
  customer_name,
  region,
  order_date,
  ship_date,
  product_id,
  category,
  sub_category,
  product_name,
  quantity,
  unit_price,
  discount,
  sales,
  profit,
  CAST(latitude AS DOUBLE)  AS latitude,
  CAST(longitude AS DOUBLE) AS longitude
FROM {FQTN}
WHERE {where_sql}
"""

# KPIs
kpis = run_query(f"""
SELECT
  SUM(sales)  AS total_sales,
  SUM(profit) AS total_profit,
  AVG(sales)  AS avg_order_value,
  COUNT(DISTINCT order_id) AS orders,
  COUNT(DISTINCT customer_id) AS customers
FROM ({base_sql}) t
""", tuple(params))

# Time series
ts = run_query(f"""
SELECT order_date, SUM(sales) AS sales, SUM(profit) AS profit
FROM ({base_sql}) t
GROUP BY order_date
ORDER BY order_date
""", tuple(params))

# Category breakdown
by_cat = run_query(f"""
SELECT category, SUM(sales) AS sales, SUM(profit) AS profit
FROM ({base_sql}) t
GROUP BY category
ORDER BY sales DESC
""", tuple(params))

# Top products
top_products = run_query(f"""
SELECT product_name, SUM(sales) AS sales, SUM(profit) AS profit
FROM ({base_sql}) t
GROUP BY product_name
ORDER BY sales DESC
LIMIT 15
""", tuple(params))

# Sample table
sample = run_query(f"""
{base_sql}
ORDER BY order_date DESC
LIMIT {limit_rows}
""", tuple(params))

# -----------------------------
# KPIs row
# -----------------------------
st.subheader("Key Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
fmt_money = lambda x: f"${x:,.0f}"
if not kpis.empty:
    c1.metric("Total Sales",   fmt_money(kpis["total_sales"][0] or 0))
    c2.metric("Total Profit",  fmt_money(kpis["total_profit"][0] or 0))
    c3.metric("Avg Order Value", fmt_money(kpis["avg_order_value"][0] or 0))
    c4.metric("Orders", int(kpis["orders"][0] or 0))
    c5.metric("Customers", int(kpis["customers"][0] or 0))
else:
    st.info("No data for current filters.")

# -----------------------------
# Charts
# -----------------------------
left, right = st.columns((3,2), gap="large")

with left:
    st.markdown("#### Sales & Profit over Time")
    if not ts.empty:
        ts_df = ts.sort_values("order_date")
        line_sales = alt.Chart(ts_df).mark_line().encode(
            x="order_date:T", y=alt.Y("sales:Q", title="Sales"), tooltip=["order_date:T","sales:Q","profit:Q"]
        )
        line_profit = alt.Chart(ts_df).mark_line(strokeDash=[4,3]).encode(
            x="order_date:T", y=alt.Y("profit:Q", title="Profit")
        )
        st.altair_chart((line_sales + line_profit).interactive(), use_container_width=True)
    else:
        st.write("—")

with right:
    st.markdown("#### Sales by Category")
    if not by_cat.empty:
        bar = alt.Chart(by_cat).mark_bar().encode(
            x=alt.X("sales:Q", title="Sales"),
            y=alt.Y("category:N", sort="-x", title="Category"),
            tooltip=["category","sales","profit"]
        )
        st.altair_chart(bar.properties(height=300), use_container_width=True)
    else:
        st.write("—")

# -----------------------------
# Analytics Wall (replaces Map)
# -----------------------------

st.markdown("### Deeper Insights")

# 1) Sales by Region over Time (stacked area)
by_region_ts = run_query(f"""
SELECT order_date, region, SUM(sales) AS sales
FROM ({base_sql}) t
GROUP BY order_date, region
ORDER BY order_date
""", tuple(params))

col1, col2 = st.columns((3,2), gap="large")

with col1:
    st.markdown("#### Sales by Region over Time")
    if not by_region_ts.empty:
        chart = alt.Chart(by_region_ts).mark_area(opacity=0.8).encode(
            x=alt.X("order_date:T", title="Date"),
            y=alt.Y("sales:Q", title="Sales"),
            color=alt.Color("region:N", title="Region"),
            tooltip=["order_date:T", "region:N", alt.Tooltip("sales:Q", format=",.2f")]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("—")

# 2) Profit Margin Distribution (histogram)
with col2:
    st.markdown("#### Profit Margin Distribution")
    # build margins safely (avoid divide-by-zero)
    margins = run_query(f"""
    SELECT
      CASE WHEN sales != 0 THEN profit / sales ELSE NULL END AS margin
    FROM ({base_sql}) t
    """, tuple(params))
    if not margins.empty:
        margins = margins.dropna()
        hist = alt.Chart(margins).mark_bar().encode(
            x=alt.X("margin:Q", bin=alt.Bin(maxbins=40), title="Profit Margin"),
            y=alt.Y("count()", title="Orders"),
            tooltip=[alt.Tooltip("count():Q", title="Orders")]
        )
        rule = alt.Chart(pd.DataFrame({"m":[float(margins["margin"].mean())]})).mark_rule().encode(x="m:Q")
        st.altair_chart((hist + rule), use_container_width=True)
    else:
        st.write("—")

st.divider()

# 3) Discount vs Profit scatter (with trend line)
st.markdown("#### Discount vs Profit (all orders)")
disc_profit = run_query(f"""
SELECT COALESCE(discount,0) AS discount, profit, sales
FROM ({base_sql}) t
""", tuple(params))

if not disc_profit.empty:
    scatter = alt.Chart(disc_profit).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X("discount:Q", title="Discount"),
        y=alt.Y("profit:Q", title="Profit"),
        tooltip=[alt.Tooltip("discount:Q", format=".2f"), alt.Tooltip("profit:Q", format=",.2f"), alt.Tooltip("sales:Q", format=",.2f")]
    )
    trend = scatter.transform_regression("discount", "profit").mark_line()
    st.altair_chart(scatter + trend, use_container_width=True)
else:
    st.write("—")

st.divider()

# 4) Pareto: Top Products by Sales with Cumulative %
st.markdown("#### Top 20 Products — Pareto View")
top_prod_full = run_query(f"""
SELECT product_name, SUM(sales) AS sales, SUM(profit) AS profit
FROM ({base_sql}) t
GROUP BY product_name
ORDER BY sales DESC
LIMIT 20
""", tuple(params))

if not top_prod_full.empty:
    dfp = top_prod_full.copy()
    dfp["rank"] = np.arange(1, len(dfp)+1)
    dfp["cum_sales"] = dfp["sales"].cumsum()
    dfp["cum_pct"] = dfp["cum_sales"] / dfp["sales"].sum()

    bars = alt.Chart(dfp).mark_bar().encode(
        x=alt.X("product_name:N", sort="-y", title="Product"),
        y=alt.Y("sales:Q", title="Sales"),
        tooltip=["product_name", alt.Tooltip("sales:Q", format=",.2f"), alt.Tooltip("profit:Q", format=",.2f")]
    ).properties(height=320)

    line = alt.Chart(dfp).mark_line(point=True).encode(
        x="product_name:N",
        y=alt.Y("cum_pct:Q", axis=alt.Axis(format="%"), title="Cumulative %"),
        tooltip=[alt.Tooltip("cum_pct:Q", title="Cum %", format=".0%")]
    )

    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), use_container_width=True)
else:
    st.write("—")

st.divider()

# 5) Category vs Sub-Category Heatmap (Sales)
st.markdown("#### Category × Sub-Category — Sales Heatmap")
heat = run_query(f"""
SELECT category, sub_category, SUM(sales) AS sales
FROM ({base_sql}) t
GROUP BY category, sub_category
""", tuple(params))

if not heat.empty:
    hm = alt.Chart(heat).mark_rect().encode(
        x=alt.X("sub_category:N", title="Sub-Category"),
        y=alt.Y("category:N", title="Category"),
        color=alt.Color("sales:Q", title="Sales", scale=alt.Scale(type="linear")),
        tooltip=["category","sub_category", alt.Tooltip("sales:Q", format=",.2f")]
    ).properties(height=260)
    st.altair_chart(hm, use_container_width=True)
else:
    st.write("—")

# -----------------------------
# Data table
# -----------------------------
st.markdown("#### Recent Orders")
st.dataframe(sample, width="stretch", hide_index=True)

# =========================================================
# AI Chat Example
# =========================================================
st.divider()
st.subheader("🤖 Demo AI Chat (restricted prompt)")

# Define canned Q → SQL mappings
CANNED = {
    "How does total sales vary month over month?": f"""
        SELECT DATE_TRUNC('month', order_date) AS month, SUM(sales) AS sales, SUM(profit) AS profit
        FROM ({base_sql}) t
        GROUP BY DATE_TRUNC('month', order_date)
        ORDER BY month
        LIMIT 120
    """,
    "What are the different product categories?": f"""
        SELECT DISTINCT category FROM ({base_sql}) t
        WHERE category IS NOT NULL
        ORDER BY category
    """,
    "What is the total profit generated from all ecommerce orders?": f"""
        SELECT SUM(profit) AS total_profit FROM ({base_sql}) t
    """,
    "Top 10 products by sales": f"""
        SELECT product_name, SUM(sales) AS sales, SUM(profit) AS profit
        FROM ({base_sql}) t
        GROUP BY product_name
        ORDER BY sales DESC
        LIMIT 10
    """,
    "Sales by region": f"""
        SELECT region, SUM(sales) AS sales, SUM(profit) AS profit
        FROM ({base_sql}) t
        GROUP BY region
        ORDER BY sales DESC
    """,
    "Average discount and margin by sub-category": f"""
        SELECT
          sub_category,
          AVG(COALESCE(discount,0)) AS avg_discount,
          AVG(CASE WHEN sales != 0 THEN profit/sales ELSE NULL END) AS avg_margin
        FROM ({base_sql}) t
        GROUP BY sub_category
        ORDER BY avg_margin DESC NULLS LAST
    """,
}

# Show quick-pick buttons
st.caption("Try one of these:")
btn_cols = st.columns(min(3, len(CANNED)))
keys = list(CANNED.keys())
for i, q in enumerate(keys):
    with btn_cols[i % len(btn_cols)]:
        if st.button(q, key=f"qbtn_{i}"):
            st.session_state.setdefault("chat", [])
            st.session_state["chat"].append(("user", q))
            st.session_state["last_q"] = q

# Free-text (will only run if it matches a canned question)
user_q = st.text_input("Or type a question exactly as listed above")
if st.button("Ask", key="ask_manual"):
    if user_q in CANNED:
        st.session_state.setdefault("chat", [])
        st.session_state["chat"].append(("user", user_q))
        st.session_state["last_q"] = user_q
    else:
        st.warning("For this demo, please click one of the suggested questions above (no LLM enabled).")

# Render chat + run SQL for the latest question
if "chat" in st.session_state and st.session_state["chat"]:
    # Display history
    for role, msg in st.session_state["chat"]:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Assistant:** {msg}")

    # If a new question was asked, answer it now
    if "last_q" in st.session_state and st.session_state["last_q"]:
        q = st.session_state["last_q"]
        sql_text = CANNED[q]
        st.markdown(f"**Assistant:** Here’s what I’d run for “_{q}_”.")
        st.code(sql_text.strip(), language="sql")
        df = run_query(sql_text, tuple(params), force_offline=force_offline)
        if df.empty:
            st.info("No rows returned for the current filters.")
        else:
            # Choose a simple viz per question
            if "month over month" in q.lower():
                ch = alt.Chart(df).mark_line().encode(
                    x="month:T",
                    y=alt.Y("sales:Q", title="Sales"),
                    tooltip=["month:T", alt.Tooltip("sales:Q", format=",.2f"),
                             alt.Tooltip("profit:Q", format=",.2f")],
                ).interactive()
                st.altair_chart(ch, use_container_width=True)
            elif q == "Sales by region":
                ch = alt.Chart(df).mark_bar().encode(
                    x=alt.X("sales:Q", title="Sales"),
                    y=alt.Y("region:N", sort="-x"),
                    tooltip=["region", alt.Tooltip("sales:Q", format=",.2f"),
                             alt.Tooltip("profit:Q", format=",.2f")],
                )
                st.altair_chart(ch, use_container_width=True)
            else:
                st.dataframe(df, width="stretch", hide_index=True)

        # Clear last_q so re-renders don't rerun unnecessarily
        st.session_state["last_q"] = ""
else:
    st.info("Click one of the suggested questions above to see the demo in action.")
