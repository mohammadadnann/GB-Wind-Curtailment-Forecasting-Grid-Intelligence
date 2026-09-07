"""GB Wind Curtailment Forecasting and Grid Intelligence Platform.

This dashboard shows day ahead forecasts for Scottish wind curtailment.
It calls the FastAPI service to get predictions and SHAP explanations for
any date in the backtest window from April 2023 to March 2026.

It was built to help battery operators and grid analysts understand when
curtailment is likely to happen, how much energy might be curtailed, and
which features are driving the forecast on any given day.
"""

import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="GB Constraint Cost Intelligence",
    layout="wide",
)

st.title("GB Wind Curtailment Forecasting and Grid Intelligence Platform")
st.caption(
    "Historical demonstration of day ahead Scottish wind curtailment forecasting. "
    "Forecasts are served from the backtest feature table, not from live data sources."
)

# Health check
try:
    health = requests.get(f"{API_BASE}/health", timeout=5).json()
    if health["status"] != "ok":
        st.error("API is not ready. Run: uvicorn src.api:app --port 8000")
        st.stop()
    available_first = health["available_dates"]["first"]
    available_last = health["available_dates"]["last"]
    st.info(
        f"Available dates: {available_first} to {available_last}. "
        f"Training cutoff: {health['training_cutoff']}."
    )
except Exception:
    st.error(
        "Cannot reach the API. Start it with: "
        "uvicorn src.api:app --host 0.0.0.0 --port 8000"
    )
    st.stop()

# Date selector
selected_date = st.date_input(
    "Select a date within the backtest window",
    value=pd.Timestamp(available_last).date(),
)

if st.button("Get forecast"):
    response = requests.get(
        f"{API_BASE}/forecast",
        params={"target_date": str(selected_date)},
        timeout=30,
    )

    if response.status_code == 404:
        st.error(response.json()["detail"])
        st.stop()

    data = response.json()
    periods = pd.DataFrame(data["periods"])
    periods["datetime_utc"] = pd.to_datetime(periods["datetime_utc"])

    st.metric("Total curtailed P50", f"{data['total_curtailed_mwh_p50']:,.0f} MWh")

    # Forecast chart
    st.subheader("Curtailment forecast with uncertainty band")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods["datetime_utc"], y=periods["curtailed_mwh_p90"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=periods["datetime_utc"], y=periods["curtailed_mwh_p10"],
        fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
        line=dict(width=0), name="P10–P90",
    ))
    fig.add_trace(go.Scatter(
        x=periods["datetime_utc"], y=periods["curtailed_mwh_p50"],
        line=dict(color="rgb(31,119,180)", width=2), name="P50",
    ))
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Curtailed MWh per period",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Constraint risk
    st.subheader("Constraint probability by period")
    fig2 = go.Figure(go.Bar(
        x=periods["datetime_utc"],
        y=periods["p_constrained"],
        marker_color=periods["p_constrained"],
        marker_colorscale="YlOrRd",
    ))
    fig2.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="P(constrained)",
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # SHAP explanation
    with st.expander("Explain a specific period"):
        period_num = st.selectbox(
            "Settlement period (1–48)",
            list(range(1, len(periods) + 1)),
        )
        explain_response = requests.get(
            f"{API_BASE}/explain",
            params={"target_date": str(selected_date), "period": period_num},
            timeout=30,
        )
        if explain_response.status_code == 200:
            explain_data = explain_response.json()
            contributions = pd.DataFrame(explain_data["top_contributions"])
            fig3 = go.Figure(go.Bar(
                x=contributions["shap_value"],
                y=contributions["feature"],
                orientation="h",
                marker_color=[
                    "crimson" if v > 0 else "steelblue"
                    for v in contributions["shap_value"]
                ],
            ))
            fig3.update_layout(
                xaxis_title="SHAP value (log scale)",
                height=400,
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Could not load explanation for this period.")

st.divider()
st.caption(
    "This is a two stage LightGBM model that first predicts whether a period will be constrained "
    "and then predicts how much curtailment will occur. "
    "Trained on Elexon balancing mechanism data and Open Meteo weather forecasts from April 2023 to March 2026. "
    "Weather forecasts are approximated from the day before at 06:00 UTC."
)