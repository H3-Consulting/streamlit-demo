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

# -----------------------------
# Page & Styles
# -----------------------------
st.set_page_config(page_title="E-Commerce Analytics (Databricks)", layout="wide")
st.title("🛍️ E-Commerce Analytics — Databricks SQL Warehouse")

st.caption("Demo dashboard reading **sandbox.ecommerce_orders** from your Databricks SQL Warehouse. \
Use the AI panel (optional) to *generate SQL from natural language* when a Model Serving endpoint is configured.")

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
def run_query(q: str, params: tuple | None = None) -> pd.DataFrame:
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                if params:
                    cur.execute(q, params)
                else:
                    cur.execute(q)
                cols = [c[0] for c in cur.description] if cur.description else []
                rows = cur.fetchall()
        return pd.DataFrame.from_records(rows, columns=cols) if cols else pd.DataFrame()
    except Exception:
        # Keep errors generic to avoid leaking details
        st.error("Query failed. Check Warehouse status/permissions and try again.")
        return pd.DataFrame()

# -----------------------------
# Schema constants (adjust if your catalog/schema differ)
# -----------------------------
CATALOG = "sandbox"
TABLE = "ecommerce_orders"   # columns from your screenshot

FQTN = f"{CATALOG}.{TABLE}"

# -----------------------------
# Load basic ranges for filters
# -----------------------------
@st.cache_data(ttl=300)
def bootstrap_filters():
    meta = run_query(f"""
        SELECT
          MIN(order_date) AS min_date,
          MAX(order_date) AS max_date
        FROM {FQTN}
    """)
    min_d = pd.to_datetime(meta["min_date"][0]).date() if not meta.empty else date(2021,1,1)
    max_d = pd.to_datetime(meta["max_date"][0]).date() if not meta.empty else date.today()
    regions = run_query(f"SELECT DISTINCT region FROM {FQTN} WHERE region IS NOT NULL ORDER BY 1")
    cats    = run_query(f"SELECT DISTINCT category FROM {FQTN} WHERE category IS NOT NULL ORDER BY 1")
    subs    = run_query(f"SELECT DISTINCT sub_category FROM {FQTN} WHERE sub_category IS NOT NULL ORDER BY 1")
    return min_d, max_d, regions["region"].tolist(), cats["category"].tolist(), subs["sub_category"].tolist()

min_d, max_d, regions, cats, subs = bootstrap_filters()

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
st.dataframe(sample, use_container_width=True, hide_index=True)

# -----------------------------
# SQL Runner (power users)
# -----------------------------
with st.expander("🔎 Run a custom SQL (safe parameters encouraged)"):
    default_sql = f"SELECT * FROM {FQTN} WHERE {where_sql} ORDER BY order_date DESC LIMIT 100"
    sql_text = st.text_area("SQL", value=default_sql, height=160)
    if st.button("Execute SQL"):
        df_custom = run_query(sql_text, tuple(params))
        st.dataframe(df_custom, use_container_width=True, hide_index=True)

# -----------------------------
# AI Generate (optional)
# -----------------------------
st.divider()
st.subheader("🤖 AI Generate (Databricks Model Serving)")

ai_endpoint = st.secrets.get("DBRICKS_AI_ENDPOINT")
ai_token = st.secrets.get("DBRICKS_AI_TOKEN")

help_text = (
    "Describe what you want (e.g., 'monthly sales and profit by region for 2024, top 10'). "
    "The assistant will propose SQL targeting the table "
    f"`{FQTN}` using its columns: order_id, customer_id, customer_name, region, order_date, ship_date, "
    "product_id, category, sub_category, product_name, quantity, unit_price, discount, sales, profit, latitude, longitude."
)

if not (ai_endpoint and ai_token):
    st.info("Configure **DBRICKS_AI_ENDPOINT** and **DBRICKS_AI_TOKEN** in Secrets to enable AI Generate.")
else:
    prompt = st.text_area("Ask in natural language", placeholder=help_text)
    cold_start = st.checkbox("Execute generated SQL automatically", value=True)
    if st.button("Generate SQL"):
        # Build a grounded prompt for SQL generation
        system = f"""
You are a helpful SQL assistant for Databricks. Produce ANSI SQL that runs on Databricks SQL Warehouse.
Use ONLY the table {FQTN} and its columns:
order_id (string), customer_id (string), customer_name (string), region (string),
order_date (date), ship_date (date), product_id (string), category (string),
sub_category (string), product_name (string), quantity (bigint), unit_price (double),
discount (double), sales (double), profit (double), latitude (string), longitude (string).
Prefer safe filters and LIMIT 500 unless user asks for a larger sample. Do not use USE statements.
Return SQL only, no explanations.
"""
        user = prompt or "Show total sales and profit by month, last 12 months."
        headers = {
            "Authorization": f"Bearer {ai_token}",
            "Content-Type": "application/json",
        }
        body = {
            # Many model serving endpoints accept 'input' or 'messages' — adapt to your endpoint.
            "messages": [
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
            ]
        }
        try:
            resp = requests.post(ai_endpoint, headers=headers, data=json.dumps(body), timeout=60)
            resp.raise_for_status()
            out = resp.json()
            # Try common response shapes
            text = (
                out.get("choices", [{}])[0].get("message", {}).get("content")
                or out.get("output_text")
                or out.get("text")
                or ""
            )
            sql_generated = text.strip().strip("```").replace("sql\n", "").replace("SQL\n", "")
            st.code(sql_generated, language="sql")

            if cold_start and sql_generated:
                st.write("Running generated SQL…")
                df_ai = run_query(sql_generated)
                if not df_ai.empty:
                    st.dataframe(df_ai, use_container_width=True, hide_index=True)
                else:
                    st.info("No rows returned.")
        except Exception:
            st.error("AI request failed. Verify your serving endpoint URL, token, and request schema.")
