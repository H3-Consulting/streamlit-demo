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
