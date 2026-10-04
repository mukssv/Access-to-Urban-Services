"""
UN-Habitat | Access to Basic Services in Cities and Urban Areas
Interactive dashboard (Streamlit + Plotly Express)

    1. Global choropleth map  : one indicator, one year, every country
    2. Progress trajectory    : ONE OR MANY places compared side by side -
                                any mix of region groupings (SDG regions and
                                continents), countries and sub-national areas
                                (regions / provinces / cities...), several
                                indicators, all years
    3. Sub-national bars      : any number of sub-national areas, several
                                indicators, one bar chart per area

----------------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------------
1. Install the required libraries (Python 3.9+):

       pip install streamlit plotly pandas

2. Put this file (app.py) and `cleaned_urban_basic_services1.csv`
   in the same folder.

3. Open a terminal IN THAT FOLDER and start the app:

       streamlit run app.py

   If Windows says "'streamlit' is not recognized", use this instead
   (it always works if the install succeeded):

       python -m pip install streamlit plotly pandas
       python -m streamlit run app.py

   The dashboard opens in your browser (default http://localhost:8501).
----------------------------------------------------------------------
"""
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
NO_DATA_GREY = "#d3d3d3"                  # colour for countries without a value
MAP_COLOR_SCALES = ["Viridis", "Plasma", "YlGnBu", "Cividis"]
DEFAULT_MAP_METRIC = "total_improved_water_pct"
DEFAULT_TRAJECTORY_METRICS = [
    "total_improved_water_pct",
    "total_improved_sanitation_pct",
    "clean_fuel_total_clean_fuel_pct",
]
ROUNDING_TOLERANCE = 1.0                  # values in (100, 101] are float noise -> 100

# Indicators where a RISE is a bad thing (colours the change arrow red)
LOWER_IS_BETTER = {
    "type_of_main_cooking_fuel_solid_fuel_coal_lignite_charcoal_wood_straw_shrub_grass_etc_pct",
}

# Spatial hierarchy options (used internally to build each series)
SCOPE_REGION = "Continent / Regional Level"
SCOPE_COUNTRY = "Country Level"
SCOPE_CITY = "City / Sub-National Level"
MAX_PLACES_RECOMMENDED = 12               # beyond this the chart gets crowded

REGION_GROUPINGS = {                      # label shown in UI -> column in the CSV
    "SDG region": "sdg_region_name",
    "Continent": "region_name",
}
SOURCE_NATIONAL = "National records"
SOURCE_SUBNATIONAL = "Aggregate of sub-national entries"

# Label formatting
ACRONYMS = {"lpg": "LPG", "vip": "VIP"}
SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
               "is", "of", "on", "or", "the", "to", "with", "no", "not", "than"}


# --------------------------------------------------------------------------- #
# Helpers: labels
# --------------------------------------------------------------------------- #
def pretty_label(col: str) -> str:
    """'total_improved_water_pct' -> 'Total Improved Water (%)'."""
    text = col.removesuffix("_pct").replace("_", " ").strip()
    text = text.replace("don t", "don't")                      # from "don't know"
    words = []
    for i, word in enumerate(text.split()):
        if word in ACRONYMS:
            words.append(ACRONYMS[word])
        elif i > 0 and word in SMALL_WORDS:
            words.append(word)                                 # keep small words lower-case
        else:
            words.append(word.capitalize())
    return " ".join(words) + " (%)"


def short_label(label: str, width: int = 90) -> str:
    """Truncate very long indicator names (e.g. the 'access to basic ...' ones)."""
    return textwrap.shorten(label, width=width, placeholder="…")


# --------------------------------------------------------------------------- #
# Data loading & preparation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data…")
def load_data(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load the full CSV (national AND sub-national records).

    Returns
    -------
    df : every record, with out-of-range percentages set to NaN
    invalid_counts : {metric_column: number of values removed because they
                      fell outside the valid 0-100 % range}
    """
    # keep_default_na=False so Namibia's ISO-2 code "NA" isn't read as missing
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["classification"] = df["classification"].astype(str).str.strip()

    # Boolean flags (robust to them being read as text)
    for flag in ["ldc", "lldc", "sids"]:
        if flag in df.columns:
            df[flag] = df[flag].astype(str).str.lower().eq("true")

    # Unique-ish label for sub-national units, e.g. "Badakhshan (Region)".
    # The same name can appear under different administrative types.
    df["entity_label"] = df["region_city_name"].astype(str) + " (" + df["classification"] + ")"

    # Percentage validation: shares must lie in [0, 100].
    #  - values only marginally above 100 are floating-point noise -> 100
    #  - anything else outside [0, 100] is a data error -> NaN
    metric_cols = [c for c in df.columns if c.endswith("_pct")]
    vals = df[metric_cols]
    vals = vals.mask(vals.gt(100) & vals.le(100 + ROUNDING_TOLERANCE), 100.0)
    invalid = vals.lt(0) | vals.gt(100)
    df[metric_cols] = vals.mask(invalid)

    return df, invalid.sum().astype(int).to_dict()


def list_metrics(df: pd.DataFrame) -> list[str]:
    """All indicator columns (suffix `_pct`)."""
    return [c for c in df.columns if c.endswith("_pct")]


def get_national(df: pd.DataFrame) -> pd.DataFrame:
    """National-level records only (used by the map)."""
    return df[df["classification"] == "National"]


# --------------------------------------------------------------------------- #
# PART 1 - MAP: data helpers
# --------------------------------------------------------------------------- #
def years_with_data(national: pd.DataFrame, metric: str, base: str) -> list[int]:
    """Years in which at least one country has a value for this metric/basis."""
    sub = national[national["base"] == base]
    return sorted(int(y) for y in sub.loc[sub[metric].notna(), "year"].unique())


def build_country_frame(national: pd.DataFrame, metric: str, base: str, year: int) -> pd.DataFrame:
    """
    One row per country for the chosen metric / basis / year.

    A few countries have two national records for the same year and basis
    (e.g. two surveys); those are averaged so every country appears once.
    """
    sub = national[(national["year"] == year) & (national["base"] == base)]
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
    """Summary numbers for the map's KPI cards (countries with a value only)."""
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
# PART 1 - MAP: figure
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
    )
    return fig


# --------------------------------------------------------------------------- #
# PART 2 - TRAJECTORY: data helpers
# --------------------------------------------------------------------------- #
LONG_COLUMNS = ["year", "metric", "value", "n", "n_label"]


def _n_label(n: int, unit: tuple[str, str] | None) -> str:
    """Hover note such as 'Based on 12 countries' (blank when not applicable)."""
    if unit is None:
        return ""
    singular, plural = unit
    return f"Based on {n} {singular if n == 1 else plural}"


def aggregate_by_year(rows: pd.DataFrame, metrics: list[str], agg: str,
                      unit: tuple[str, str] | None) -> pd.DataFrame:
    """
    Collapse records to one value per year and metric.

    agg  : 'mean' or 'median' (a single record per year is returned unchanged)
    unit : (singular, plural) noun for the hover note, or None

    Returns a long table: year, metric, value, n (records behind the value), n_label.
    """
    if rows.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    grouped = rows.groupby("year")[metrics]
    values = (getattr(grouped, agg)().reset_index()
              .melt(id_vars="year", var_name="metric", value_name="value"))
    counts = (grouped.count().reset_index()
              .melt(id_vars="year", var_name="metric", value_name="n"))

    long = values.merge(counts, on=["year", "metric"]).dropna(subset=["value"])
    long["year"] = long["year"].astype(int)
    long["n_label"] = [_n_label(int(n), unit) for n in long["n"]]
    return long.reset_index(drop=True)


def build_trajectory(df: pd.DataFrame, *, scope: str, group_col: str, region: str,
                     country: str | None, entity: str | None, country_source: str,
                     agg: str, base: str, metrics: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Build the long-format time series for the selected place.

    Regional  : national value of each country -> mean/median across the region's countries
    Country   : the country's national records, or (alternatively / if it has
                none) the mean/median of its sub-national entries
    City      : the selected sub-national unit's own records
    """
    rows_base = df[df["base"] == base]
    meta = {"fallback": False}

    if scope == SCOPE_REGION:
        rows = rows_base[(rows_base["classification"] == "National")
                         & (rows_base[group_col] == region)]
        # One value per country-year first, so a country with two surveys in the
        # same year does not get double weight in the regional mean/median.
        rows = rows.groupby(["country_name", "year"], as_index=False)[metrics].mean()
        long = aggregate_by_year(rows, metrics, agg, ("country", "countries"))

    elif scope == SCOPE_COUNTRY:
        in_country = rows_base[rows_base["country_name"] == country]
        national_rows = in_country[in_country["classification"] == "National"]
        if country_source == SOURCE_NATIONAL and not national_rows.empty:
            # Two national surveys in one year are simply averaged
            long = aggregate_by_year(national_rows, metrics, "mean", None)
        else:
            # e.g. Argentina has no national records in this dataset
            meta["fallback"] = country_source == SOURCE_NATIONAL
            sub_rows = in_country[in_country["classification"] != "National"]
            long = aggregate_by_year(sub_rows, metrics, agg,
                                     ("sub-national entry", "sub-national entries"))

    else:  # SCOPE_CITY
        rows = rows_base[(rows_base["country_name"] == country)
                         & (rows_base["classification"] != "National")
                         & (rows_base["entity_label"] == entity)]
        long = aggregate_by_year(rows, metrics, "mean", None)   # duplicates averaged

    return long, meta


def net_changes(long: pd.DataFrame, metrics: list[str]) -> dict:
    """
    Net change in percentage points from the earliest to the latest year with data.
    `delta` is None when only one year is available; the entry is None when no data.
    """
    out = {}
    for metric in metrics:
        series = long[long["metric"] == metric].sort_values("year")
        if series.empty:
            out[metric] = None
            continue
        first, last = series.iloc[0], series.iloc[-1]
        out[metric] = {
            "y0": int(first["year"]), "v0": float(first["value"]),
            "y1": int(last["year"]), "v1": float(last["value"]),
            "delta": float(last["value"] - first["value"]) if len(series) > 1 else None,
        }
    return out


# --------------------------------------------------------------------------- #
# PART 2 - TRAJECTORY: figure
# --------------------------------------------------------------------------- #
def make_trajectory_chart(long: pd.DataFrame, metrics: list[str],
                          label_map: dict, color_map: dict) -> go.Figure:
    """Multi-line chart: one coloured, toggleable line per selected indicator."""
    plot = long.copy()
    plot["indicator"] = plot["metric"].map(lambda m: short_label(label_map[m], 60))
    plot["indicator_hover"] = plot["metric"].map(
        lambda m: "<br>".join(textwrap.wrap(label_map[m], 45)))
    plot = plot.sort_values(["indicator", "year"])

    order = [short_label(label_map[m], 60) for m in metrics]
    colors = {short_label(label_map[m], 60): color_map[m] for m in metrics}

    fig = px.line(
        plot, x="year", y="value", color="indicator", markers=True,
        custom_data=["indicator_hover", "n_label"],
        color_discrete_map=colors,
        category_orders={"indicator": order},
    )
    fig.update_traces(
        line=dict(width=2.5), marker=dict(size=8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Year: %{x}<br>"
            "Access rate: <b>%{y:.1f}%</b><br>"
            "%{customdata[1]}"
            "<extra></extra>"
        ),
    )

    # Dashed reference line: SDG target of universal (100 %) access
    fig.add_hline(
        y=100, line_dash="dash", line_color="#6b7280", line_width=1.5,
        annotation_text="SDG target: 100%", annotation_position="top left",
    )

    year_span = int(plot["year"].max() - plot["year"].min())
    grid = "rgba(128,128,128,0.25)"
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(title_text="", orientation="h", yanchor="top", y=-0.2, x=0),
        xaxis=dict(title="Year", tickformat="d", showgrid=True, gridcolor=grid,
                   dtick=1 if year_span <= 12 else None),
        # Ticks stop at 100 %; the axis runs to 105 so the 100 % line and
        # markers sitting on it are not clipped at the plot edge.
        yaxis=dict(title="Access rate (%)", range=[0, 105],
                   tickvals=list(range(0, 101, 20)), ticksuffix="%",
                   showgrid=True, gridcolor=grid, zeroline=False),
    )
    return fig


def show_plot(fig: go.Figure, key: str | None = None) -> None:
    """Render a figure full-width (handles old and new Streamlit APIs)."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key)
    except TypeError:                      # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


# --------------------------------------------------------------------------- #
# PART 1 - MAP: page section
# --------------------------------------------------------------------------- #
def render_map_section(df: pd.DataFrame, metric_cols: list[str], label_map: dict,
                       invalid_counts: dict, base: str) -> None:
    national = get_national(df)

    # ------------------------------ Sidebar ------------------------------- #
    st.sidebar.subheader("🗺️ World map")
    default_idx = metric_cols.index(DEFAULT_MAP_METRIC) if DEFAULT_MAP_METRIC in metric_cols else 0
    metric = st.sidebar.selectbox("Map indicator", metric_cols, index=default_idx,
                                  format_func=label_map.get)
    only_with_data = st.sidebar.checkbox(
        "Only list years with data", value=True,
        help="Survey coverage is sparse; untick to see every year from the dataset.",
    )

    all_years = sorted(int(y) for y in national["year"].dropna().unique())
    year_options = years_with_data(national, metric, base) if only_with_data else all_years
    if not year_options:
        st.subheader("🗺️ World map")
        st.warning("No national data are available for this indicator and data basis.")
        return
    if len(year_options) == 1:
        year = year_options[0]
        st.sidebar.info(f"Only {year} is available.")
    else:
        year = st.sidebar.select_slider("Map year", options=year_options, value=year_options[-1])

    palette = st.sidebar.selectbox("Color scale", MAP_COLOR_SCALES)

    if invalid_counts.get(metric, 0):
        st.sidebar.caption(
            f"⚠️ {invalid_counts[metric]} value(s) outside 0–100% were excluded "
            f"for this indicator (across all records)."
        )

    # --------------------------- Data for the map -------------------------- #
    label = label_map[metric]
    frame = build_country_frame(national, metric, base, year)
    reporting = frame.dropna(subset=["value"])

    # Countries known to the dataset but without a value for this selection
    universe = national.drop_duplicates("iso_alpha3")[["iso_alpha3", "country_name"]]
    no_data = universe[~universe["iso_alpha3"].isin(reporting["iso_alpha3"])]

    # ------------------------------ Title + KPIs --------------------------- #
    st.subheader(f"🗺️ {short_label(label)} — {year}")
    st.caption(f"Basis: {base.lower()}-level | Light grey = no data for this indicator and year.")

    kpi = compute_kpis(reporting)
    c1, c2, c3, c4 = st.columns(4)
    if kpi:
        c1.metric("Sample average (unweighted)", f"{kpi['mean']:.1f}%")
        c2.metric(f"Highest: {kpi['top_name']}", f"{kpi['top_value']:.1f}%")
        c3.metric(f"Lowest: {kpi['low_name']}", f"{kpi['low_value']:.1f}%")
        c4.metric("Countries reporting", f"{kpi['n']}")
    else:
        for col, name in zip((c1, c2, c3, c4),
                             ("Sample average", "Highest", "Lowest", "Countries reporting")):
            col.metric(name, "n/a")
        st.info("No country reports a value for this indicator in the selected year.")

    # ------------------------------- Map ----------------------------------- #
    show_plot(make_map(reporting, no_data, label, palette))

    with st.expander("Show map data table"):
        table = (
            reporting[["country_name", "sdg_region_name", "value"]]
            .sort_values("value", ascending=False)
            .rename(columns={"country_name": "Country", "sdg_region_name": "SDG region",
                             "value": short_label(label, 50)})
        )
        st.dataframe(table, hide_index=True)


# --------------------------------------------------------------------------- #
# PART 2 - TRAJECTORY: multi-place comparison
# --------------------------------------------------------------------------- #
def build_comparison(df: pd.DataFrame, places: list[dict], *, country_source: str,
                     agg: str, base: str, metrics: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Build one long table for every selected place (adds a `place` column).

    Returns (long, fallback_countries) where fallback_countries lists countries
    that have no national records and were therefore shown as an aggregate of
    their sub-national entries.
    """
    frames, fallbacks = [], []
    for p in places:
        long, meta = build_trajectory(
            df, scope=p["scope"], group_col=p.get("group_col"), region=p.get("region"),
            country=p.get("country"), entity=p.get("entity"),
            country_source=country_source, agg=agg, base=base, metrics=metrics,
        )
        if meta["fallback"]:
            fallbacks.append(p["country"])
        if not long.empty:
            frames.append(long.assign(place=p["label"], level=p["level"]))
    if not frames:
        return pd.DataFrame(columns=LONG_COLUMNS + ["place", "level"]), fallbacks
    return pd.concat(frames, ignore_index=True), fallbacks


def net_change_table(long: pd.DataFrame, places: list[str], metrics: list[str],
                     label_map: dict) -> pd.DataFrame:
    """Earliest -> latest change (percentage points) for every place x indicator."""
    rows = []
    for metric in metrics:
        for place in places:
            s = long[(long["metric"] == metric) & (long["place"] == place)].sort_values("year")
            if s.empty:
                rows.append({"Indicator": label_map[metric], "Place": place,
                             "First year": None, "First %": None,
                             "Latest year": None, "Latest %": None, "Change (pp)": None})
                continue
            first, last = s.iloc[0], s.iloc[-1]
            rows.append({
                "Indicator": label_map[metric], "Place": place,
                "First year": int(first["year"]), "First %": round(float(first["value"]), 1),
                "Latest year": int(last["year"]), "Latest %": round(float(last["value"]), 1),
                "Change (pp)": round(float(last["value"] - first["value"]), 1) if len(s) > 1 else None,
            })
    return pd.DataFrame(rows)


def make_comparison_chart(long: pd.DataFrame, metrics: list[str], place_order: list[str],
                          place_colors: dict, label_map: dict) -> go.Figure:
    """
    Compare several places: one coloured line per place.
    With several indicators the chart is split into one panel per indicator.
    """
    plot = long.copy()
    wrap = lambda m: "<br>".join(textwrap.wrap(label_map[m], 45))
    plot["indicator"] = plot["metric"].map(wrap)
    indicator_order = [wrap(m) for m in metrics]
    plot = plot.sort_values(["place", "year"])

    multi = len(metrics) > 1
    n_rows = -(-len(metrics) // 2) if multi else 1
    fig = px.line(
        plot, x="year", y="value", color="place", markers=True,
        facet_col="indicator" if multi else None,
        facet_col_wrap=2 if multi else 0,
        facet_col_spacing=0.07, facet_row_spacing=0.14,
        custom_data=["indicator", "level", "n_label"],
        color_discrete_map=place_colors,
        category_orders={"place": place_order, "indicator": indicator_order},
    )
    fig.update_traces(
        line=dict(width=2.5), marker=dict(size=7),
        hovertemplate=(
            "<b>%{fullData.name}</b> (%{customdata[1]})<br>"
            "%{customdata[0]}<br>"
            "Year: %{x}<br>"
            "Access rate: <b>%{y:.1f}%</b><br>"
            "%{customdata[2]}"
            "<extra></extra>"
        ),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=", 1)[-1]))

    fig.add_hline(y=100, line_dash="dash", line_color="#6b7280", line_width=1.2)

    grid = "rgba(128,128,128,0.25)"
    year_span = int(plot["year"].max() - plot["year"].min())
    fig.update_xaxes(title_text="Year", tickformat="d", showgrid=True, gridcolor=grid,
                     showticklabels=True, dtick=1 if year_span <= 12 else None)
    fig.update_yaxes(title_text="", range=[0, 105], tickvals=list(range(0, 101, 20)),
                     ticksuffix="%", showgrid=True, gridcolor=grid, zeroline=False,
                     showticklabels=True)
    fig.update_layout(
        height=480 if not multi else 330 * n_rows + 120,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(title_text="", orientation="h", yanchor="top",
                    y=-0.12 if not multi else -0.06, x=0),
    )
    return fig


# --------------------------------------------------------------------------- #
# PART 2 - TRAJECTORY: page section
# --------------------------------------------------------------------------- #
def render_trajectory_section(df: pd.DataFrame, metric_cols: list[str], label_map: dict,
                              color_map: dict, base: str) -> None:
    # ------------------------------ Sidebar ------------------------------- #
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Progress trajectory")
    st.sidebar.caption("Pick any mix of places below - they are all drawn on the same chart "
                       "so you can compare them.")

    # ---- (a) region groupings & continents (several groupings can be mixed) ----
    region_options = []                      # [(grouping label, region name), ...]
    for grouping, col in REGION_GROUPINGS.items():
        region_options += [(grouping, name) for name in sorted(df[col].dropna().unique())]
    default_regions = [o for o in region_options if o[0] == "Continent"][:2]
    sel_regions = st.sidebar.multiselect(
        "Regions & continents", region_options, default=default_regions,
        format_func=lambda o: f"{o[1]}  ·  {o[0]}",
        help="'SDG region' is the UN SDG regional grouping; 'Continent' uses the UN M49 "
             "continents. You can select several of either kind.",
    )

    # ---- (b) countries ----
    all_countries = sorted(df["country_name"].dropna().unique())
    sel_countries = st.sidebar.multiselect(
        "Countries", all_countries,
        help="Type to search. Each country becomes its own line.",
    )
    country_source = SOURCE_NATIONAL
    if sel_countries:
        country_source = st.sidebar.radio(
            "Country data source", [SOURCE_NATIONAL, SOURCE_SUBNATIONAL],
            help="National records are used as reported. The alternative aggregates every "
                 "sub-national entry (regions, provinces, districts, cities…) per year.",
        )

    # ---- (c) sub-national areas / cities ----
    sub_all = df[(df["classification"] != "National") & (df["base"] == base)]
    sub_countries = sorted(sub_all["country_name"].dropna().unique())
    narrow = st.sidebar.multiselect(
        "Sub-national areas: narrow by country", sub_countries,
        help="Optional. Restricts the list below to these countries. Leave empty to search "
             "every country.",
    )
    pool = sub_all[sub_all["country_name"].isin(narrow)] if narrow else sub_all
    entity_options = sorted(
        pool[["country_name", "entity_label"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda t: (t[1], t[0]),
    )
    sel_entities = st.sidebar.multiselect(
        "Sub-national areas / cities", entity_options,
        format_func=lambda t: f"{t[1]}, {t[0]}",
        help="Regions, provinces, districts, cities… (type to search). "
             "Only units with records on the chosen data basis are listed.",
    )

    # ---- aggregation (only matters for aggregated series) ----
    agg = "mean"
    if sel_regions or sel_countries:
        agg = st.sidebar.radio(
            "Aggregation", ["Mean", "Median"], horizontal=True,
            help="How several values in one year are combined (countries within a region, "
                 "or sub-national entries within a country).",
        ).lower()

    defaults = [m for m in DEFAULT_TRAJECTORY_METRICS if m in metric_cols]
    metrics = st.sidebar.multiselect(
        "Service indicators", metric_cols, default=defaults, format_func=label_map.get,
        help="Select as many as you like.",
    )

    # ------------------------- Assemble selected places -------------------- #
    places: list[dict] = []
    for grouping, name in sel_regions:
        places.append({"label": f"{name} ({grouping})", "level": grouping,
                       "scope": SCOPE_REGION, "group_col": REGION_GROUPINGS[grouping],
                       "region": name})
    for c in sel_countries:
        places.append({"label": c, "level": "Country", "scope": SCOPE_COUNTRY, "country": c})
    for country, entity in sel_entities:
        places.append({"label": f"{entity}, {country}", "level": "Sub-national",
                       "scope": SCOPE_CITY, "country": country, "entity": entity})

    # ----------------------------- Page header ----------------------------- #
    st.subheader("📈 Progress trajectory")
    if not places:
        st.info("Select at least one region, country or sub-national area in the sidebar.")
        return
    if not metrics:
        st.info("Select at least one service indicator in the sidebar.")
        return
    if len(places) > MAX_PLACES_RECOMMENDED:
        st.warning(f"{len(places)} places are selected - the chart may be hard to read. "
                   f"Consider comparing {MAX_PLACES_RECOMMENDED} or fewer.")

    # ------------------------------- Data ---------------------------------- #
    long, fallbacks = build_comparison(
        df, places, country_source=country_source, agg=agg, base=base, metrics=metrics)

    selected_labels = [p["label"] for p in places]
    if long.empty:
        st.info(f"No data for the selected place(s) on the {base.lower()} basis "
                f"for the selected indicators.")
        return
    reporting = [l for l in selected_labels if l in set(long["place"])]
    missing = [l for l in selected_labels if l not in reporting]
    if missing:
        st.warning("No data on this basis / indicator selection for: " + "; ".join(missing))

    year_min, year_max = int(long["year"].min()), int(long["year"].max())
    place_colors = {l: px.colors.qualitative.Dark24[i % 24] for i, l in enumerate(selected_labels)}

    # ------------------------------ Title & notes -------------------------- #
    if len(reporting) == 1:
        st.markdown(f"### Progress Trajectory of Urban Services for {reporting[0]} "
                    f"({year_min}–{year_max})")
    else:
        st.markdown(f"### Comparing {len(reporting)} places ({year_min}–{year_max})")

    notes = []
    if sel_regions:
        notes.append(f"regions/continents = {agg} of national values across their countries")
    if sel_countries:
        notes.append("countries = national survey records" if country_source == SOURCE_NATIONAL
                     else f"countries = {agg} of all sub-national entries")
    if sel_entities:
        notes.append("sub-national areas = their own records")
    st.caption(f"{base.lower()}-level data · " + "; ".join(notes)
               + " · lines connect the years that have data · dashed line = SDG target (100%).")
    if fallbacks:
        st.info(f"{', '.join(sorted(set(fallbacks)))} "
                f"{'has' if len(set(fallbacks)) == 1 else 'have'} no national records in this "
                f"dataset, so the {agg} of sub-national entries is shown instead.")
    if sel_regions:
        st.caption("⚠️ The set of reporting countries changes from year to year, so part of a "
                   "regional trend can reflect who was surveyed, not real change. "
                   "Hover over a point to see how many countries it is based on.")

    # ------------------------------- Chart --------------------------------- #
    if len(reporting) == 1:
        # Single place: indicators are the lines (original view) + KPI cards
        single = long[long["place"] == reporting[0]]
        changes = net_changes(single, metrics)
        for start in range(0, len(metrics), 4):
            cols = st.columns(4)
            for col, metric in zip(cols, metrics[start:start + 4]):
                info = changes[metric]
                name = short_label(label_map[metric], 40)
                if info is None:
                    col.metric(name, "n/a", help=f"{label_map[metric]}: no data for this selection.")
                elif info["delta"] is None:
                    col.metric(f"{name} ({info['y1']})", f"{info['v1']:.1f}%",
                               help=f"{label_map[metric]}: only one year of data, so no change can be shown.")
                else:
                    col.metric(
                        f"{name} ({info['y0']}–{info['y1']})", f"{info['v1']:.1f}%",
                        delta=f"{info['delta']:+.1f} pp",
                        delta_color="inverse" if metric in LOWER_IS_BETTER else "normal",
                        help=(f"{label_map[metric]}: {info['v0']:.1f}% in {info['y0']} → "
                              f"{info['v1']:.1f}% in {info['y1']}. "
                              f"'pp' = percentage points (net change, earliest to latest year)."),
                    )
        show_plot(make_trajectory_chart(single, metrics, label_map, color_map))
    else:
        show_plot(make_comparison_chart(long, metrics, reporting, place_colors, label_map))

        st.markdown("**Net change, earliest → latest year with data** "
                    "(pp = percentage points)")
        change_tbl = net_change_table(long, reporting, metrics, label_map)
        st.dataframe(change_tbl, hide_index=True)

    with st.expander("Show trajectory data table"):
        table = (
            long.assign(Indicator=long["metric"].map(label_map))
            .pivot_table(index="year", columns=["Indicator", "place"], values="value")
            .sort_index().round(1)
        )
        if len(reporting) == 1:
            table.columns = table.columns.droplevel("place")
        st.dataframe(table)


# --------------------------------------------------------------------------- #
# PART 3 - SUB-NATIONAL BAR CHARTS
# --------------------------------------------------------------------------- #
LATEST = "Latest available year"


def build_subnational_snapshot(df: pd.DataFrame, areas: list[tuple[str, str]], metrics: list[str],
                               base: str, year: int | None) -> pd.DataFrame:
    """
    One value per (area, indicator).

    year=None -> each indicator uses the most recent year in which that area
                 reports it (years can therefore differ between indicators).
    year=int  -> only that year; indicators without a value stay NaN.
    Duplicate records for the same area and year are averaged.
    """
    rows_out = []
    for country, entity in areas:
        rows = df[(df["base"] == base) & (df["country_name"] == country)
                  & (df["classification"] != "National") & (df["entity_label"] == entity)]
        by_year = rows.groupby("year")[metrics].mean()
        for metric in metrics:
            series = by_year[metric].dropna() if not by_year.empty else pd.Series(dtype=float)
            if year is None:
                value, yr = (float(series.iloc[-1]), int(series.index[-1])) if not series.empty \
                    else (float("nan"), None)
            else:
                value = float(series.get(year, float("nan")))
                yr = year if value == value else None
            rows_out.append({"area": f"{entity}, {country}", "metric": metric,
                             "value": value, "year": yr})
    return pd.DataFrame(rows_out)


def make_area_bar_chart(snapshot: pd.DataFrame, area: str, metrics: list[str],
                        label_map: dict, color_map: dict) -> go.Figure:
    """Horizontal bar chart of the selected indicators for one sub-national area."""
    sub = snapshot[snapshot["area"] == area].set_index("metric").reindex(metrics).reset_index()
    sub["indicator"] = sub["metric"].map(lambda m: short_label(label_map[m], 45))
    sub["indicator_hover"] = sub["metric"].map(lambda m: "<br>".join(textwrap.wrap(label_map[m], 45)))
    has = sub["value"].notna()
    sub["plot_value"] = sub["value"].fillna(0)             # missing -> empty bar
    sub["text"] = [f"{v:.1f}%" + (f" ({int(y)})" if pd.notna(y) else "") if ok else "no data"
                   for v, y, ok in zip(sub["value"], sub["year"], has)]
    sub["shown"] = [f"<b>{v:.1f}%</b>" + (f" in {int(y)}" if pd.notna(y) else "") if ok
                    else "No data" for v, y, ok in zip(sub["value"], sub["year"], has)]

    fig = px.bar(
        sub, x="plot_value", y="indicator", orientation="h", text="text",
        color="indicator",
        color_discrete_map={short_label(label_map[m], 45): color_map[m] for m in metrics},
        category_orders={"indicator": [short_label(label_map[m], 45) for m in metrics]},
        custom_data=["indicator_hover", "shown"],
    )
    fig.update_traces(
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    )
    grid = "rgba(128,128,128,0.25)"
    fig.update_layout(
        title=dict(text=f"<b>{area}</b>", x=0, font=dict(size=15)),
        height=max(220, 70 + 55 * len(metrics)),
        margin=dict(l=10, r=60, t=45, b=10),
        showlegend=False, bargap=0.35,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Access rate (%)", range=[0, 100], ticksuffix="%",
                   tickvals=list(range(0, 101, 20)), showgrid=True, gridcolor=grid),
        yaxis=dict(title="", autorange="reversed"),          # first indicator on top
    )
    return fig


def render_subnational_bars_section(df: pd.DataFrame, metric_cols: list[str], label_map: dict,
                                    color_map: dict, base: str) -> None:
    # ------------------------------ Sidebar ------------------------------- #
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Sub-national comparison (bars)")

    sub_all = df[(df["classification"] != "National") & (df["base"] == base)]
    narrow = st.sidebar.multiselect(
        "Narrow areas by country", sorted(sub_all["country_name"].dropna().unique()),
        key="bar_narrow",
        help="Optional. Restricts the list below to these countries.",
    )
    pool = sub_all[sub_all["country_name"].isin(narrow)] if narrow else sub_all
    area_options = sorted(
        pool[["country_name", "entity_label"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda t: (t[1], t[0]),
    )
    areas = st.sidebar.multiselect(
        "Sub-national areas", area_options, key="bar_areas",
        format_func=lambda t: f"{t[1]}, {t[0]}",
        help="Regions, provinces, districts, cities… Select any number; each gets its own bar chart.",
    )
    defaults = [m for m in DEFAULT_TRAJECTORY_METRICS if m in metric_cols]
    metrics = st.sidebar.multiselect(
        "Indicators to compare", metric_cols, default=defaults, key="bar_metrics",
        format_func=label_map.get,
    )

    # Years in which any chosen area reports any chosen indicator
    years = []
    if areas and metrics:
        keys = {(c, e) for c, e in areas}
        hit = sub_all[[(c, e) in keys for c, e in zip(sub_all["country_name"], sub_all["entity_label"])]]
        years = sorted((int(y) for y in hit.loc[hit[metrics].notna().any(axis=1), "year"].unique()),
                       reverse=True)
    year_choice = st.sidebar.selectbox(
        "Year", [LATEST] + years, key="bar_year",
        help="'Latest available year' uses each indicator's most recent value for each area "
             "(the year is shown on the bar). Pick a year to compare all areas in the same year.",
    )
    year = None if year_choice == LATEST else int(year_choice)

    # ----------------------------- Page header ----------------------------- #
    st.subheader("📊 Sub-national areas: indicator comparison")
    if not areas:
        st.info("Select one or more sub-national areas in the sidebar.")
        return
    if not metrics:
        st.info("Select at least one indicator in the sidebar.")
        return

    snapshot = build_subnational_snapshot(df, areas, metrics, base, year)
    st.caption(f"{base.lower()}-level data · "
               + ("most recent value per indicator (year in brackets)" if year is None
                  else f"values for {year}")
               + " · one panel per area, shared 0–100% scale.")

    # ------------------------------- Charts -------------------------------- #
    area_labels = [f"{e}, {c}" for c, e in areas]
    per_row = 1 if len(metrics) > 6 else 2
    for start in range(0, len(area_labels), per_row):
        cols = st.columns(per_row)
        for col, area in zip(cols, area_labels[start:start + per_row]):
            with col:
                if snapshot.loc[snapshot["area"] == area, "value"].notna().any():
                    show_plot(make_area_bar_chart(snapshot, area, metrics, label_map, color_map),
                              key=f"bar_chart_{area}")
                else:
                    st.markdown(f"**{area}**")
                    st.info("No data for the selected indicators"
                            + ("" if year is None else f" in {year}") + ".")

    with st.expander("Show comparison data table"):
        table = (
            snapshot.assign(Indicator=snapshot["metric"].map(label_map))
            .pivot(index="Indicator", columns="area", values="value")
            .reindex([label_map[m] for m in metrics])
            .round(1)
        )
        st.dataframe(table)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Urban Basic Services", layout="wide")
    st.markdown(
        """
        <style>
            html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stApp {
                font-family: Georgia, serif !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Access to Basic Services in Cities and Urban Areas")
    st.caption("Source: UN-Habitat Global Urban Indicators Database.")

    # Locate the CSV next to this script first, then in the working directory
    data_path = Path(__file__).resolve().parent / DATA_FILE
    if not data_path.exists():
        data_path = Path(DATA_FILE)
    if not data_path.exists():
        st.error(f"Could not find `{DATA_FILE}`. Place it in the same folder as app.py.")
        st.stop()

    df, invalid_counts = load_data(str(data_path))

    # Dictionary: raw column name -> friendly display name
    metric_cols = list_metrics(df)
    label_map = {col: pretty_label(col) for col in metric_cols}

    # One stable colour per indicator, so a line keeps its colour when the
    # selection changes.
    palette = px.colors.qualitative.Dark24
    color_map = {col: palette[i % len(palette)] for i, col in enumerate(metric_cols)}

    # Shared control: the data basis applies to both the map and the chart
    st.sidebar.header("Controls")
    bases = sorted(df["base"].dropna().unique())
    base = st.sidebar.radio(
        "Data basis (map and chart)", bases,
        help="Results are reported either at household level or at population level.",
    )

    # Page order: map first, trajectory chart directly below it
    render_map_section(df, metric_cols, label_map, invalid_counts, base)
    st.markdown("---")
    render_trajectory_section(df, metric_cols, label_map, color_map, base)
    st.markdown("---")
    render_subnational_bars_section(df, metric_cols, label_map, color_map, base)


if __name__ == "__main__":
    main()
