from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_FILE = "cleaned_urban_basic_services1.csv"
NO_DATA_GREY = "#d3d3d3"                 # colour for countries without a value
COLOR_SCALES = ["Viridis", "Plasma", "YlGnBu", "Cividis"]
DEFAULT_METRIC = "total_improved_water_pct"
ROUNDING_TOLERANCE = 1.0                 # values in (100, 101] are float noise -> 100
ACRONYMS = {"lpg": "LPG", "vip": "VIP"}  # nicer capitalisation in labels


# --------------------------------------------------------------------------- #
# Helpers: labels
# --------------------------------------------------------------------------- #
def pretty_label(col: str) -> str:
    """'total_improved_sanitation_pct' -> 'Total improved sanitation (%)'."""
    text = col.removesuffix("_pct").replace("_", " ").strip()
    text = text.replace("don t", "don't")                      # from "don't know"
    text = text.capitalize()
    for short, full in ACRONYMS.items():
        text = re.sub(rf"\b{short}\b", full, text, flags=re.IGNORECASE)
    return f"{text} (%)"


def short_label(label: str, width: int = 90) -> str:
    """Truncate very long indicator names (e.g. the 'access to basic ...' ones)."""
    return textwrap.shorten(label, width=width, placeholder="…")


# --------------------------------------------------------------------------- #
# Data loading & preparation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data…")
def load_national_data(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load the CSV and keep national-level records only.

    Returns
    -------
    df : national-level rows, with out-of-range percentages set to NaN
    invalid_counts : {metric_column: number of values removed because they
                      fell outside the valid 0-100 % range}
    """
    # keep_default_na=False so Namibia's ISO-2 code "NA" isn't read as missing
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])

    # 1) National records only
    df = df[df["classification"].astype(str).str.strip().eq("National")].copy()

    # 2) Boolean flags (robust to them being read as text)
    for flag in ["ldc", "lldc", "sids"]:
        if flag in df.columns:
            df[flag] = df[flag].astype(str).str.lower().eq("true")

    # 3) Percentage validation. Shares must lie in [0, 100]:
    #    - values only marginally above 100 are floating-point noise -> 100
    #    - anything else outside [0, 100] is a data error -> NaN
    metric_cols = [c for c in df.columns if c.endswith("_pct")]
    vals = df[metric_cols]
    vals = vals.mask(vals.gt(100) & vals.le(100 + ROUNDING_TOLERANCE), 100.0)
    invalid = vals.lt(0) | vals.gt(100)
    df[metric_cols] = vals.mask(invalid)

    return df, invalid.sum().astype(int).to_dict()


def list_metrics(df: pd.DataFrame) -> list[str]:
    """All indicator columns (suffix `_pct`)."""
    return [c for c in df.columns if c.endswith("_pct")]


def years_with_data(df: pd.DataFrame, metric: str, base: str) -> list[int]:
    """Years in which at least one country has a value for this metric/basis."""
    sub = df[df["base"] == base]
    return sorted(int(y) for y in sub.loc[sub[metric].notna(), "year"].unique())


def build_country_frame(df: pd.DataFrame, metric: str, base: str, year: int) -> pd.DataFrame:
    """
    One row per country for the chosen metric / basis / year.

    A few countries have two national records for the same year and basis
    (e.g. two surveys); those are averaged so every country appears once.
    """
    sub = df[(df["year"] == year) & (df["base"] == base)]
    frame = (
        sub.groupby("iso_alpha3", as_index=False)
        .agg(
            country_name=("country_name", "first"),
            sdg_region_name=("sdg_region_name", "first"),
            ldc=("ldc", "max"),
            sids=("sids", "max"),
            value=(metric, "mean"),          # NaN if every record is missing
        )
    )
    frame["sdg_region_name"] = frame["sdg_region_name"].fillna("n/a")
    frame["status"] = (
        "LDC: " + frame["ldc"].map({True: "Yes", False: "No"})
        + " | SIDS: " + frame["sids"].map({True: "Yes", False: "No"})
    )
    return frame


def compute_kpis(reporting: pd.DataFrame) -> dict:
    """Summary numbers for the KPI cards (countries with a value only)."""
    if reporting.empty:
        return {}
    top = reporting.loc[reporting["value"].idxmax()]
    low = reporting.loc[reporting["value"].idxmin()]
    return {
        "mean": reporting["value"].mean(),   # simple (unweighted) country average
        "top_name": top["country_name"], "top_value": top["value"],
        "low_name": low["country_name"], "low_value": low["value"],
        "n": len(reporting),
    }


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
def make_map(reporting: pd.DataFrame, no_data: pd.DataFrame, label: str, palette: str) -> go.Figure:
    """Choropleth: coloured countries = data, light grey = no value."""
    hover_label = "<br>".join(textwrap.wrap(label, 40))
    hover_template = (
        "<b>%{customdata[0]}</b><br>"
        "SDG region: %{customdata[1]}<br>"
        f"{hover_label}:<br>"
        "<b>%{z:.1f}%</b><br>"
        "%{customdata[2]}"
        "<extra></extra>"
    )

    if reporting.empty:
        fig = go.Figure()
    else:
        fig = px.choropleth(
            reporting,
            locations="iso_alpha3",            # ISO-3 codes
            locationmode="ISO-3",
            color="value",
            custom_data=["country_name", "sdg_region_name", "status"],
            color_continuous_scale=palette,
            range_color=(0, 100),              # fixed 0-100 % scale
        )
        fig.update_traces(hovertemplate=hover_template, marker_line_width=0.4,
                          marker_line_color="white")

    # Countries in the dataset without a value for this metric/year: grey trace
    # so they stay hoverable ("No data") instead of silently disappearing.
    if not no_data.empty:
        fig.add_trace(go.Choropleth(
            locations=no_data["iso_alpha3"],
            locationmode="ISO-3",
            z=[0] * len(no_data),
            colorscale=[[0, NO_DATA_GREY], [1, NO_DATA_GREY]],
            showscale=False,
            customdata=no_data[["country_name"]].to_numpy(),
            hovertemplate="<b>%{customdata[0]}</b><br>No data for this indicator/year<extra></extra>",
            marker_line_width=0.4,
            marker_line_color="white",
        ))

    # Land colour is grey, so any country without a value (including ones that
    # are not in the dataset at all) is drawn light grey rather than blank.
    fig.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=True, coastlinecolor="#9aa5b1",
        showland=True, landcolor=NO_DATA_GREY,
        showcountries=True, countrycolor="white",
        showocean=True, oceancolor="#eaf3fa",
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title="%", ticksuffix="%", len=0.7),
        font=dict(family="Georgia, Times New Roman, serif"),
    )
    return fig


def show_plot(fig: go.Figure) -> None:
    """Render the figure full-width (handles old and new Streamlit APIs)."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None)
    except TypeError:                      # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, theme=None)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Urban Basic Services", page_icon="🌍", layout="wide")
    st.markdown(
        """
        <style>
            html, body, [class*="st"], .stApp, .stSidebar, .stTabs,
            .stDataFrame, .stTable, .stMarkdown, .stCaption,
            div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
                font-family: Georgia, "Times New Roman", serif !important;
            }
            h1, h2, h3, h4, h5, h6 {
                font-family: Georgia, "Times New Roman", serif !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Access to Basic Services in Cities and Urban Areas")
    st.caption("Source: UN-Habitat Global Urban Indicators Database (national-level records).")

    # Locate the CSV next to this script first, then in the working directory
    data_path = Path(__file__).resolve().parent / DATA_FILE
    if not data_path.exists():
        data_path = Path(DATA_FILE)
    if not data_path.exists():
        st.error(f"Could not find `{DATA_FILE}`. Place it in the same folder as app.py.")
        st.stop()

    df, invalid_counts = load_national_data(str(data_path))
    metric_cols = list_metrics(df)
    all_years = sorted(int(y) for y in df["year"].dropna().unique())
    bases = sorted(df["base"].dropna().unique())

    # ------------------------------ Sidebar ------------------------------- #
    st.sidebar.header("Controls")

    default_idx = metric_cols.index(DEFAULT_METRIC) if DEFAULT_METRIC in metric_cols else 0
    metric = st.sidebar.selectbox(
        "Indicator", metric_cols, index=default_idx, format_func=pretty_label
    )
    base = st.sidebar.radio(
        "Data basis", bases,
        help="Results are reported either at household level or at population level.",
    )
    only_with_data = st.sidebar.checkbox(
        "Only list years with data", value=True,
        help="Survey coverage is sparse; untick to see every year from the dataset.",
    )

    year_options = years_with_data(df, metric, base) if only_with_data else all_years
    if not year_options:
        st.warning("No national data are available for this indicator and data basis.")
        st.stop()
    if len(year_options) == 1:
        year = year_options[0]
        st.sidebar.info(f"Only {year} is available.")
    else:
        year = st.sidebar.select_slider("Year", options=year_options, value=year_options[-1])

    palette = st.sidebar.selectbox("Color scale", COLOR_SCALES)

    if invalid_counts.get(metric, 0):
        st.sidebar.caption(
            f"⚠️ {invalid_counts[metric]} value(s) outside 0–100% were excluded for this indicator."
        )

    # --------------------------- Data for the map -------------------------- #
    label = pretty_label(metric)
    frame = build_country_frame(df, metric, base, year)
    reporting = frame.dropna(subset=["value"])

    # Countries known to the dataset but without a value for this selection
    universe = df.drop_duplicates("iso_alpha3")[["iso_alpha3", "country_name"]]
    no_data = universe[~universe["iso_alpha3"].isin(reporting["iso_alpha3"])]

    # ------------------------------ Title + KPIs --------------------------- #
    st.subheader(f"{short_label(label)} — {year}")

    # kpi = compute_kpis(reporting)
    # c1, c2, c3, c4 = st.columns(4)
    # if kpi:
    #     c1.metric("Sample average (unweighted)", f"{kpi['mean']:.1f}%")
    #     c2.metric(f"Highest: {kpi['top_name']}", f"{kpi['top_value']:.1f}%")
    #     c3.metric(f"Lowest: {kpi['low_name']}", f"{kpi['low_value']:.1f}%")
    #     c4.metric("Countries reporting", f"{kpi['n']}")
    # else:
    #     for col, name in zip((c1, c2, c3, c4),
    #                          ("Sample average", "Highest", "Lowest", "Countries reporting")):
    #         col.metric(name, "n/a")
    #     st.info("No country reports a value for this indicator in the selected year.")

    # ------------------------------- Map ----------------------------------- #
    show_plot(make_map(reporting, no_data, label, palette))

    with st.expander("Show data table"):
        table = (
            reporting[["country_name", "sdg_region_name", "value"]]
            .sort_values("value", ascending=False)
            .rename(columns={"country_name": "Country", "sdg_region_name": "SDG region",
                             "value": f"{short_label(label, 50)}"})
        )
        st.dataframe(table, hide_index=True)


if __name__ == "__main__":
    main()
