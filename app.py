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

# =========================================================
# Power-user SQL runner (restricted to sandbox.ecommerce_orders + auto-injected filters)
# =========================================================
with st.expander("🔎 Run a custom SQL (restricted to sandbox.ecommerce_orders & current filters)"):

    default_sql = """
    SELECT
      order_id, customer_id, customer_name, region, order_date, ship_date,
      product_id, category, sub_category, product_name, quantity, unit_price,
      discount, sales, profit
    FROM sandbox.ecommerce_orders
    ORDER BY order_date DESC
    LIMIT 100
    """.strip()

    user_sql = st.text_area("SQL (READ-ONLY)", value=default_sql, height=160)

    # ---------- helpers ----------
    def _sql_clean_upper(s: str) -> str:
        return " ".join(s.upper().replace("\n", " ").split())

    def _quote(v: str) -> str:
        # simple literal escaper for controlled sidebar inputs
        return "'" + str(v).replace("'", "''") + "'"

    def build_filters_sql_literal() -> str:
        parts = [f"order_date BETWEEN {_quote(start_date)} AND {_quote(end_date)}"]
        if region:
            parts.append("region IN (" + ", ".join(_quote(r) for r in region) + ")")
        if category:
            parts.append("category IN (" + ", ".join(_quote(c) for c in category) + ")")
        if subcat:
            parts.append("sub_category IN (" + ", ".join(_quote(s) for s in subcat) + ")")
        return " AND ".join(parts)

    def is_safe_and_scoped(sql_text: str) -> tuple[bool, str]:
        """
        Enforce:
          - single statement (no ';')
          - no comments (/* */ or --)
          - no JOIN/UNION/; (keeps scope on single table)
          - only reads (no INSERT/UPDATE/DELETE/MERGE/DDL/GRANT/REVOKE)
          - must reference SANDBOX.ECOMMERCE_ORDERS (optionally with alias)
        Return (ok, reason_if_not_ok)
        """
        u = _sql_clean_upper(sql_text)

        if ";" in sql_text:
            return False, "Multiple statements (';') are not allowed."
        if "/*" in sql_text or "--" in sql_text:
            return False, "SQL comments are not allowed."
        banned = ("INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE")
        if any(f" {kw} " in f" {u} " for kw in banned):
            return False, "Write/DDL statements are not allowed."
        if " JOIN " in u or " UNION " in u:
            return False, "JOIN/UNION are not allowed in the demo runner."
        # must target the single allowed table
        if "SANDBOX.ECOMMERCE_ORDERS" not in u:
            return False, "Query must reference sandbox.ecommerce_orders."
        return True, ""

    def inject_filters(sql_text: str, filters_sql: str) -> str:
        """
        Append current filters to the user's query.
        If there's already a WHERE, add 'AND (filters)'. Else add 'WHERE (filters)'.
        We append before any trailing whitespace; we don't try to insert before ORDER/GROUP.
        """
        s = sql_text.strip().rstrip(";").strip()
        u = _sql_clean_upper(s)
        if " WHERE " in u:
            return f"{s} AND ({filters_sql})"
        else:
            return f"{s} WHERE ({filters_sql})"

    # ---------- run ----------
    if st.button("Execute SQL", key="run_custom_sql_restricted"):
        ok, reason = is_safe_and_scoped(user_sql)
        if not ok:
            st.error(f"❌ Query not allowed: {reason}")
        else:
            filters_sql = build_filters_sql_literal()
            final_sql = inject_filters(user_sql, filters_sql)
            st.code(final_sql, language="sql")
            # Note: we intentionally do NOT pass params; filters are already inlined safely
            df_custom = run_query(final_sql)
            if df_custom.empty:
                st.info("No rows returned for the current filters.")
            else:
                st.dataframe(df_custom, use_container_width=True, hide_index=True)

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
        df = run_query(sql_text, tuple(params))
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
                st.dataframe(df, use_container_width=True, hide_index=True)

        # Clear last_q so re-renders don't rerun unnecessarily
        st.session_state["last_q"] = ""
else:
    st.info("Click one of the suggested questions above to see the demo in action.")
