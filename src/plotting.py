import os
import sys

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Band, Button, ColumnDataSource, Legend, Select, Span
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

from logger import Logger


def load_logger():
    idx = min(i for i in range(len(sys.argv)) if "plotting.py" in sys.argv[i])
    if sys.argv[idx + 1] == "--args":
        idx += 1
    filenames = sys.argv[idx + 1 :]

    loggers = []
    for filename in filenames:
        loggers.append(Logger.load(filename))
    if len(loggers) == 1:
        logger = loggers[0]
    else:
        logger = Logger()
        for other, filename in zip(loggers, filenames):
            for k, v in other.data.items():
                idx = k.find(".")
                key_head, key_tail = k[:idx], k[idx:]
                new_key = f"{key_head}/{os.path.split(filename)[1]}{key_tail}"
                logger.data[new_key] = v
    return logger


def get_data(
    query: str, center_style: str, ci_style: str
) -> dict[str, list[np.ndarray]]:
    assert center_style in ("Mean", "Median")
    assert ci_style in ("None", "STD", "25% / 75%", "5% / 95%", "2.5% / 97.5%")

    logger = load_logger()

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
        "name": selected,
    }


def main():
    logger = load_logger()

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
        options=["None", "STD", "25% / 75%", "5% / 95%", "2.5% / 97.5%"],
    )
    refresh_button = Button(
        label="Refresh", button_type="primary", height=32, width=64
    )

    plot = figure(
        tools="crosshair,pan,reset,save,wheel_zoom,box_zoom",
        x_range=(0, 1),
        y_range=(0, 1),
    )

    plot.add_layout(
        Span(location=0, dimension="height", line_color="black", line_width=2)
    )
    plot.add_layout(
        Span(location=0, dimension="width", line_color="black", line_width=2)
    )

    def update(_attr, _old, _new):
        data = get_data(
            metric_input.value,
            center_style_select.value,
            ci_style_select.value,
        )
        if not data:
            return

        ymin = min(np.quantile(arr, 0.1) for arr in data["lower"])
        ymax = max(np.quantile(arr, 0.9) for arr in data["upper"])
        if ymin == ymax:
            if ymin == ymax == 0:
                ymin = -1
                ymax = 1
            else:
                ymin = min(ymin, 2 * ymin, 0)
                ymax = max(ymax, 2 * ymax, 0)

        plot.x_range.start = -0.2
        plot.x_range.end = max(len(arr) for arr in data["y"]) - 0.8

        plot.y_range.start = ymin - (ymax - ymin) * 0.2
        plot.y_range.end = ymax + (ymax - ymin) * 0.2

        plot.renderers = []
        plot.center = [
            item
            for item in plot.center
            if not isinstance(item, Band) and not isinstance(item, Legend)
        ]

        for i in range(len(data["x"])):
            plot.line(
                x=data["x"][i],
                y=data["y"][i],
                color=data["color"][i],
                legend_label=data["name"][i],
                line_width=4,
            )
            band = Band(
                base="x",
                lower="lower",
                upper="upper",
                fill_alpha=0.2,
                fill_color=data["color"][i],
                source=ColumnDataSource(
                    {
                        "x": data["x"][i],
                        "lower": data["lower"][i],
                        "upper": data["upper"][i],
                    }
                ),
            )
            plot.circle(
                x=data["x"][i], y=data["y"][i], color=data["color"][i], size=6
            )
            plot.add_layout(band)

    metric_input.on_change("value", update)
    center_style_select.on_change("value", update)
    ci_style_select.on_change("value", update)
    refresh_button.on_event("button_click", lambda _: update(None, None, None))

    update(None, None, None)

    curdoc().add_root(
        column(
            row(
                metric_input,
                center_style_select,
                ci_style_select,
            ),
            refresh_button,
            plot,
        )
    )


main()
