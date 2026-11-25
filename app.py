import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION (Gestalt: Figure-Ground & Prägnanz / Simplicity)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KMD Synoptic Weather Dashboard | Gestalt Visual Analytics",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS enforcing Common Region, Figure-Ground, and Proximity
st.markdown("""
<style>
    .stApp {
        background-color: #F1F5F9;
    }
    .dashboard-header {
        background-color: #FFFFFF;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        margin-bottom: 1rem;
    }
    .dashboard-header h1 {
        color: #0F172A;
        font-size: 1.65rem;
        margin: 0 0 0.25rem 0;
    }
    .dashboard-header p {
        color: #475569;
        font-size: 0.92rem;
        margin: 0;
    }
    .kpi-card {
        background-color: #FFFFFF;
        padding: 1rem 1.15rem;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.15rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# Okabe-Ito Colorblind-Safe Palette (Gestalt: Similarity across every chart)
STATION_COLORS = {
    "Mombasa": "#D55E00",
    "Kisumu":  "#0072B2",
    "Nairobi": "#009E73",
    "Nyeri":   "#CC79A7",
    "Eldoret": "#E69F00"
}
STATION_ORDER = ["Mombasa", "Kisumu", "Nairobi", "Nyeri", "Eldoret"]

# ---------------------------------------------------------------------------
# DATA LOADING & CLEANING PIPELINE
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Station", "Date"]).reset_index(drop=True)
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")

    # Domain-consistent imputation for missing records
    df.loc[df["Rainfall_mm"].isnull() & (df["Weather_Condition"] == "Mostly Dry"), "Rainfall_mm"] = 0.0
    for col in ["Rainfall_mm", "Relative_Humidity_pct", "Wind_Speed_kmh"]:
        df[col] = df[col].fillna(df.groupby(["Station", "Month"])[col].transform("median"))
    return df

try:
    df = load_and_clean_data("Kenya_Meteorological_Synthetic_Dataset.csv")
except FileNotFoundError:
    df = load_and_clean_data("Kenya_Meteorological_Cleaned.csv")

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS (Gestalt: Common Region & Proximity of Controls)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌦️ KMD Weather Observatory")
    st.caption("Gestalt Perceptual Visual Analytics")
    st.divider()

    st.markdown("### Station & Temporal Filters")
    selected_stations = st.multiselect(
        "Select Meteorological Stations",
        options=STATION_ORDER,
        default=STATION_ORDER,
        help="Filter observations across Kenya's five synoptic weather stations."
    )

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input(
        "Observation Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    show_gestalt_notes = st.toggle("Show Gestalt Design Annotations", value=True)
    st.divider()

    st.markdown("### Gestalt Principles Applied")
    st.markdown("""
    - **Common Region:** Bounded card panels & sidebar isolation
    - **Similarity:** Consistent station color encoding across all views
    - **Proximity:** Co-located KPI summaries & paired rainfall metrics
    - **Continuity:** Unbroken monthly temperature trajectories
    - **Closure:** Clean open-frame axes & unified box/violin enclosures
    - **Figure-Ground:** High-contrast data marks on soft slate canvas
    - **Connectedness:** Linked line markers & station regime centroids
    - **Prägnanz:** Minimal gridlines, zero 3D distortion, clear hierarchy
    """)

if not selected_stations:
    st.warning("Please select at least one meteorological station in the sidebar.")
    st.stop()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_dt, end_dt = pd.to_datetime(min_date), pd.to_datetime(max_date)

filtered_df = df[
    (df["Station"].isin(selected_stations)) &
    (df["Date"] >= start_dt) &
    (df["Date"] <= end_dt)
].copy()

if filtered_df.empty:
    st.warning("No observations match the selected filter criteria.")
    st.stop()

# ---------------------------------------------------------------------------
# HEADER & EXECUTIVE KPI ROW (Gestalt: Visual Hierarchy & Proximity)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="dashboard-header">
    <h1>Kenya Meteorological Department — Synoptic Weather Intelligence Dashboard</h1>
    <p>Interactive Multi-Station Climate Monitoring ({", ".join(selected_stations)}) &nbsp;|&nbsp;
       Period: <b>{start_dt.strftime('%d %b %Y')}</b> to <b>{end_dt.strftime('%d %b %Y')}</b> &nbsp;|&nbsp;
       Records: <b>{len(filtered_df):,}</b></p>
</div>
""", unsafe_allow_html=True)

station_means = filtered_df.groupby("Station")["Mean_Temperature_C"].mean()
warmest_st = station_means.idxmax()
warmest_val = station_means.max()
peak_max_temp = filtered_df["Maximum_Temperature_C"].max()

station_rain = filtered_df.groupby("Station")["Rainfall_mm"].sum()
wettest_st = station_rain.idxmax()
wettest_val = station_rain.max()
wettest_days = int((filtered_df[filtered_df["Station"] == wettest_st]["Rainfall_mm"] > 0).sum())

mean_rh = filtered_df["Relative_Humidity_pct"].mean()
min_rh_st = filtered_df.groupby("Station")["Relative_Humidity_pct"].mean().min()
max_rh_st = filtered_df.groupby("Station")["Relative_Humidity_pct"].mean().max()

station_wind = filtered_df.groupby("Station")["Wind_Speed_kmh"].mean()
windiest_st = station_wind.idxmax()
windiest_val = station_wind.max()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 5px solid {STATION_COLORS.get(warmest_st, '#D55E00')};">
        <div class="kpi-title">Warmest Station (Mean)</div>
        <div class="kpi-value">{warmest_st} ({warmest_val:.2f}°C)</div>
        <div class="kpi-sub">Absolute Peak Max: {peak_max_temp:.1f}°C</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 5px solid {STATION_COLORS.get(wettest_st, '#0072B2')};">
        <div class="kpi-title">Wettest Station (Total)</div>
        <div class="kpi-value">{wettest_st} ({wettest_val:,.1f} mm)</div>
        <div class="kpi-sub">{wettest_days} Rainy Days in Period</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 5px solid #009E73;">
        <div class="kpi-title">Mean Relative Humidity</div>
        <div class="kpi-value">{mean_rh:.1f}%</div>
        <div class="kpi-sub">Station Means: {min_rh_st:.1f}% – {max_rh_st:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 5px solid {STATION_COLORS.get(windiest_st, '#CC79A7')};">
        <div class="kpi-title">Highest Wind Regime</div>
        <div class="kpi-value">{windiest_st} ({windiest_val:.2f} km/h)</div>
        <div class="kpi-sub">Max Gust/Daily: {filtered_df['Wind_Speed_kmh'].max():.1f} km/h</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ROW 1: TEMPERATURE COMPARISON & RAINFALL DISTRIBUTION
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.subheader("1. Temperature Comparison Across Stations")
        active_order = [s for s in STATION_ORDER if s in selected_stations]
        fig_temp = px.box(
            filtered_df,
            x="Station",
            y="Mean_Temperature_C",
            color="Station",
            category_orders={"Station": active_order},
            color_discrete_map=STATION_COLORS,
            points="outliers",
            labels={"Mean_Temperature_C": "Daily Mean Temperature (°C)", "Station": "Meteorological Station"}
        )
        # Overlay station means for direct quantitative comparison
        mean_pts = filtered_df.groupby("Station")["Mean_Temperature_C"].mean().reindex(active_order)
        fig_temp.add_trace(go.Scatter(
            x=active_order,
            y=mean_pts.values,
            mode="markers+text",
            marker=dict(color="#0F172A", size=9, symbol="diamond"),
            text=[f"{v:.2f}°C" for v in mean_pts.values],
            textposition="top center",
            name="Station Mean"
        ))
        fig_temp.update_layout(
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#FFFFFF",
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
            height=380
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        if show_gestalt_notes:
            st.caption("Gestalt Note (Closure & Common Region): Bounded box-and-whisker shapes enclose quartile spreads while a unified card container separates thermal distributions from precipitation metrics.")

with col_right:
    with st.container(border=True):
        st.subheader("2. Rainfall Distribution Across Stations")
        rain_summary = filtered_df.groupby("Station").agg(
            Total_Rainfall=("Rainfall_mm", "sum"),
            Rainy_Days=("Rainfall_mm", lambda x: (x > 0).sum())
        ).reindex(active_order).reset_index()

        fig_rain = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Cumulative Rainfall (mm)", "Rainy Days (> 0 mm)"),
            horizontal_spacing=0.14
        )
        fig_rain.add_trace(
            go.Bar(
                x=rain_summary["Station"],
                y=rain_summary["Total_Rainfall"],
                marker_color=[STATION_COLORS[s] for s in rain_summary["Station"]],
                text=[f"{v:,.0f}" for v in rain_summary["Total_Rainfall"]],
                textposition="outside",
                name="Total (mm)"
            ),
            row=1, col=1
        )
        fig_rain.add_trace(
            go.Bar(
                x=rain_summary["Station"],
                y=rain_summary["Rainy_Days"],
                marker_color=[STATION_COLORS[s] for s in rain_summary["Station"]],
                marker_opacity=0.82,
                text=[f"{int(v)}d" for v in rain_summary["Rainy_Days"]],
                textposition="outside",
                name="Rainy Days"
            ),
            row=1, col=2
        )
        fig_rain.update_yaxes(range=[0, rain_summary["Total_Rainfall"].max() * 1.18], row=1, col=1)
        fig_rain.update_yaxes(range=[0, rain_summary["Rainy_Days"].max() * 1.20], row=1, col=2)
        fig_rain.update_layout(
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#FFFFFF",
            margin=dict(t=35, b=20, l=20, r=20),
            showlegend=False,
            height=380
        )
        st.plotly_chart(fig_rain, use_container_width=True)
        if show_gestalt_notes:
            st.caption("Gestalt Note (Proximity & Similarity): Placing cumulative volume and rainy-day frequency side-by-side in matching station colors enables instant differentiation between heavy intensity (Kisumu) and high frequency (Mombasa).")

# ---------------------------------------------------------------------------
# ROW 2: MONTHLY TREND, HUMIDITY VS RAINFALL & WIND SPEED
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.subheader("3. Monthly Temperature Trend")
        monthly_df = filtered_df.groupby(["Month", "Month_Name", "Station"])["Mean_Temperature_C"].mean().reset_index()
        monthly_df = monthly_df.sort_values("Month")
        fig_trend = px.line(
            monthly_df,
            x="Month_Name",
            y="Mean_Temperature_C",
            color="Station",
            markers=True,
            category_orders={"Station": active_order},
            color_discrete_map=STATION_COLORS,
            labels={"Month_Name": "Month", "Mean_Temperature_C": "Mean Temp (°C)"}
        )
        fig_trend.update_traces(line=dict(width=2.5), marker=dict(size=6))
        fig_trend.update_layout(
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#FFFFFF",
            margin=dict(t=20, b=20, l=15, r=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
            height=360
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        if show_gestalt_notes:
            st.caption("Gestalt Note (Continuity & Connectedness): Continuous lines connecting monthly nodes guide the eye along seasonal warming (Mar–May) and cooling (Jul–Oct) arcs.")

with c2:
    with st.container(border=True):
        st.subheader("4. Humidity vs. Rainfall")
        fig_scat = px.scatter(
            filtered_df,
            x="Relative_Humidity_pct",
            y="Rainfall_mm",
            color="Station",
            opacity=0.45,
            category_orders={"Station": active_order},
            color_discrete_map=STATION_COLORS,
            hover_data=["Date", "Weather_Condition"],
            labels={"Relative_Humidity_pct": "Relative Humidity (%)", "Rainfall_mm": "Rainfall (mm)"}
        )
        # Station regime centroids
        cent = filtered_df.groupby("Station")[["Relative_Humidity_pct", "Rainfall_mm"]].mean().reindex(active_order).reset_index()
        fig_scat.add_trace(go.Scatter(
            x=cent["Relative_Humidity_pct"],
            y=cent["Rainfall_mm"],
            mode="markers",
            marker=dict(
                size=13,
                symbol="diamond",
                color=[STATION_COLORS[s] for s in cent["Station"]],
                line=dict(width=1.8, color="#0F172A")
            ),
            name="Station Centroid"
        ))
        fig_scat.update_layout(
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#FFFFFF",
            margin=dict(t=20, b=20, l=15, r=15),
            showlegend=False,
            height=360
        )
        st.plotly_chart(fig_scat, use_container_width=True)
        if show_gestalt_notes:
            st.caption("Gestalt Note (Figure-Ground & Similarity): Translucent daily points recede into the ground while bold outlined diamonds foreground each station's mean moisture-rainfall regime.")

with c3:
    with st.container(border=True):
        st.subheader("5. Wind Speed Comparison")
        fig_wind = px.violin(
            filtered_df,
            x="Station",
            y="Wind_Speed_kmh",
            color="Station",
            box=True,
            points=False,
            category_orders={"Station": active_order},
            color_discrete_map=STATION_COLORS,
            labels={"Wind_Speed_kmh": "Wind Speed (km/h)", "Station": "Station"}
        )
        fig_wind.update_layout(
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#FFFFFF",
            margin=dict(t=20, b=20, l=15, r=15),
            showlegend=False,
            height=360
        )
        st.plotly_chart(fig_wind, use_container_width=True)
        if show_gestalt_notes:
            st.caption("Gestalt Note (Prägnanz / Simplicity): Symmetric violin silhouettes with embedded quartile boxes communicate coastal vs. highland wind regimes without visual clutter.")

# ---------------------------------------------------------------------------
# ROW 3: SYNOPTIC DATA EXPLORER & EXPORT
# ---------------------------------------------------------------------------
with st.expander("Inspect Cleaned Synoptic Dataset & Summary Statistics", expanded=False):
    tab1, tab2 = st.tabs(["Station Summary Table", "Filtered Daily Records"])
    with tab1:
        summary_tbl = filtered_df.groupby("Station").agg(
            Observations=("Date", "count"),
            Mean_Temp_C=("Mean_Temperature_C", "mean"),
            Min_Temp_C=("Minimum_Temperature_C", "min"),
            Max_Temp_C=("Maximum_Temperature_C", "max"),
            Total_Rainfall_mm=("Rainfall_mm", "sum"),
            Rainy_Days=("Rainfall_mm", lambda s: (s > 0).sum()),
            Mean_Humidity_pct=("Relative_Humidity_pct", "mean"),
            Mean_Wind_kmh=("Wind_Speed_kmh", "mean"),
            Mean_Pressure_hPa=("Atmospheric_Pressure_hPa", "mean")
        ).round(2)
        st.dataframe(summary_tbl, use_container_width=True)
    with tab2:
        st.dataframe(filtered_df.drop(columns=["Month", "Month_Name"]), use_container_width=True, height=280)
        csv_bytes = filtered_df.drop(columns=["Month", "Month_Name"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Cleaned Dataset (CSV)",
            data=csv_bytes,
            file_name="Kenya_Meteorological_Filtered.csv",
            mime="text/csv"
        )
