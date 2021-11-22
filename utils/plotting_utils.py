# flake8: noqa
import altair as alt
import pandas as pd

from data_utils import determine_k_unit_str
from process_analytics.defaults import (
    TIME_TITLE,
    K_TITLE,
    HEIGHT,
    WIDTH,
    START_TOOLTIP_TITLE,
    DURATION_TOOLTIP_TITLE,
    TIME_FORMAT,
    I_TITLE,
)


def process_bar_chart(dataset: pd.DataFrame, agg_demand_unit: str):
    """Stacked bar chart for active processes."""
    k_unit_str = determine_k_unit_str(agg_demand_unit)
    chart_specs = (
        alt.Chart(dataset)
        .mark_bar()
        .encode(
            x=alt.X("previous_belief_time:T", title=TIME_TITLE),
            x2="belief_time:T",
            y=alt.Y("sum(k):Q", title=f"{K_TITLE} ({k_unit_str})"),
            color=alt.Color(
                "floor_i:O", legend=None, scale=alt.Scale(scheme="tableau20")
            ),
            order=alt.Order("d:Q", sort="descending"),
            tooltip=[
                alt.Tooltip("full_start_date:N", title=START_TOOLTIP_TITLE),
                alt.Tooltip("d:Q", title=DURATION_TOOLTIP_TITLE),
                alt.Tooltip("sum(k):Q", title=f"{K_TITLE} ({k_unit_str})"),
                alt.Tooltip("i:O", title=I_TITLE),
            ],
        )
    )
    chart_specs["height"] = HEIGHT
    chart_specs["width"] = WIDTH
    chart_specs["transform"] = [
        {"as": "full_start_date", "calculate": f"timeFormat(datum.s, '{TIME_FORMAT}')"},
        {"as": "floor_i", "calculate": "floor(datum.i)"},
    ]
    return chart_specs


def save_html(html: str, file: str):
    """Save html string to file."""
    with open(file, "w") as f:
        f.write(html)


def from_html_template(
    chart: alt.Chart, connection: str, demand_label: str = "water"
) -> str:
    """Embed Altair chart in html template (string)."""
    vega_version = alt.VEGA_VERSION
    vegalite_version = alt.VEGALITE_VERSION
    vegaembed_version = alt.VEGAEMBED_VERSION

    spec = chart.to_json(indent=None)
    title = f"""
        {demand_label.capitalize()} demand at {connection} - 
        a description of underlying processes (based on meter data only)
    """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <script src="https://cdn.jsdelivr.net/npm/vega@{vega_version}"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite_version}"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed_version}"></script>
    </head>
    <body>

        <h2>{title}</h2>
        <div id="vis"></div>

        <script type="text/javascript">
            vegaEmbed('#vis', {spec}).catch(console.error);
        </script>
    </body>
    </html>
    """


def altairify(df: pd.DataFrame) -> pd.DataFrame:
    """Altair likes dates represented as integers."""
    for dt_col in ["belief_time", "s", "e"]:
        df[dt_col] = df[dt_col].astype("int64") // 1e6
    return df
