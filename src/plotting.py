import sys

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    Legend,
    LegendItem,
    MultiLine,
    Select,
    Span,
)
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

from logger import Logger


def get_data(
    logger: Logger, query: str, center_style: str, ci_style: str
) -> dict[str, list[np.ndarray]]:
    assert center_style in ("Mean", "Median")
    assert ci_style in ("STD", "25% / 75%", "5% / 95%", "2.5% / 97.5%")

    selected = [
        k.replace(".data", "")
        for k in sorted(logger.data.keys())
        if k.startswith(query) and k.endswith(".data")
    ]
    if len(selected) == 0:
        raise ValueError

    centers: list[np.ndarray] = []
    for k in selected:
        if (
            center_style == "Median"
            and f"{k}.quantile_0.5" in logger.data.keys()
        ):
            centers.append(np.asarray(logger.data[f"{k}.quantile_0.5"]))
        else:
            assert center_style == "Mean"
            centers.append(np.asarray(logger.data[f"{k}.data"]))
    xs = [np.arange(len(y)) for y in centers]

    lower = []
    upper = []
    for i, k in enumerate(selected):
        if ci_style == "STD" and f"{k}.std" in logger.data.keys():
            lower.append(centers[i] - np.asarray(logger.data[f"{k}.std"]))
            upper.append(centers[i] + np.asarray(logger.data[f"{k}.std"]))
        elif (
            ci_style == "25% / 75%"
            and f"{k}.quantile_0.25" in logger.data.keys()
        ):
            lower.append(np.asarray(logger.data[f"{k}.quantile_0.25"]))
            upper.append(np.asarray(logger.data[f"{k}.quantile_0.75"]))
        elif (
            ci_style == "5% / 95%"
            and f"{k}.quantile_0.05" in logger.data.keys()
        ):
            lower.append(np.asarray(logger.data[f"{k}.quantile_0.05"]))
            upper.append(np.asarray(logger.data[f"{k}.quantile_0.95"]))
        elif (
            ci_style == "2.5% / 97.5%"
            and f"{k}.quantile_0.025" in logger.data.keys()
        ):
            lower.append(np.asarray(logger.data[f"{k}.quantile_0.025"]))
            upper.append(np.asarray(logger.data[f"{k}.quantile_0.975"]))
        else:
            lower.append(centers[i])
            upper.append(centers[i])

    palette = np.asarray(Category10_10)
    return {
        "x": xs,
        "y": centers,
        "lower": lower,
        "upper": upper,
        "color": np.asarray(palette)[np.arange(len(xs)) % len(palette)],
    }


def main():
    filename = sys.argv[1]

    logger = Logger.load(filename)

    metrics = set()
    for k in logger.data.keys():
        if not k.endswith(".data"):
            continue
        subkey = k[: -len(".data")]
        path = subkey.split("/")
        for i in range(1, len(path) + 1):
            if i == len(path):
                metrics.add(subkey)
            else:
                metrics.add("/".join(path[:i]) + "/")

    metrics = sorted(list(metrics))
    metric_input = Select(title="Metric", value=metrics[0], options=metrics)
    center_style_select = Select(
        title="Center Style",
        value="Mean",
        options=["Mean", "Median"],
    )
    ci_style_select = Select(
        title="Lower/Upper Bound Style",
        value="STD",
        options=["STD", "25% / 75%", "5% / 95%", "2.5% / 97.5%"],
    )

    source = ColumnDataSource(
        get_data(
            logger,
            metrics[0],
            center_style_select.value,
            ci_style_select.value,
        )
    )

    plot = figure(
        tools="crosshair,pan,reset,save,wheel_zoom,box_zoom",
        x_range=(0, 1),
        y_range=(0, 1),
    )
    # hover = plot.select({"type": HoverTool})
    # hover.tooltips = [("epoch", "@x"), ("value", "@y")]

    # plot.line(x="x", y="y", line_width=3, source=source)
    center_glyph = MultiLine(xs="x", ys="y", line_width=4, line_color="color")
    lower_glyph = MultiLine(
        xs="x", ys="lower", line_width=2, line_color="color"
    )
    upper_glyph = MultiLine(
        xs="x", ys="upper", line_width=2, line_color="color"
    )

    plot.add_glyph(source, center_glyph)
    plot.add_glyph(source, lower_glyph)
    plot.add_glyph(source, upper_glyph)
    # hover.renderers = [center_renderer]

    plot.circle(x="x", y="y", size=6, source=source)
    plot.add_layout(
        Span(location=0, dimension="height", line_color="black", line_width=2)
    )
    plot.add_layout(
        Span(location=0, dimension="width", line_color="black", line_width=2)
    )

    def update(_attr, _old, _new):
        data = get_data(
            logger,
            metric_input.value,
            center_style_select.value,
            ci_style_select.value,
        )
        if not data:
            return

        ymin = min(arr.min() for arr in data["lower"])
        ymax = max(arr.max() for arr in data["upper"])

        plot.x_range.start = -0.2
        plot.x_range.end = len(data["y"][0]) - 0.8

        plot.y_range.start = ymin - (ymax - ymin) * 0.1
        plot.y_range.end = ymax + (ymax - ymin) * 0.1

        source.data = data

    metric_input.on_change("value", update)
    center_style_select.on_change("value", update)
    ci_style_select.on_change("value", update)
    update(None, None, None)

    curdoc().add_root(
        column(metric_input, center_style_select, ci_style_select, plot)
    )


main()
