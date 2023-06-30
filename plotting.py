import sys

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (
    Band,
    ColumnDataSource,
    HoverTool,
    MultiLine,
    Select,
    Span,
    TextInput,
)
from bokeh.plotting import figure

from logger import Logger

for i, entry in enumerate(sys.argv):
    if entry in ("-f", "--file"):
        filename = sys.argv[i + 1]

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
print(metrics)

metrics = sorted(list(metrics))
metric_input = Select(title="Metric", value=metrics[0], options=metrics)
style_select = Select(
    title="Style",
    value="mean + std",
    options=["mean", "mean + std", "median + quantiles"],
)


# def update(_attrname, _old, _new):
def get_data():
    query, style = metric_input.value, style_select.value

    selected = [
        k.replace(".data", "")
        for k in sorted(logger.data.keys)
        if k.endswith(".data")
    ]
    if len(selected) == 0:
        raise ValueError

    if style == "median + quantiles":
        ys = [
            np.asarray(
                logger.data[f"{k}.quantile_0.5"]
                if f"{k}.quantile_0.5" in logger.data
                else None
            )
            for k in selected
        ]
        los = [np.asarray(logger.data[f"{k}.quantile_0.25"])]
    else:
        ys = [np.asarray(logger.data[f"{k}.data"]) for k in selected]
        xs = [np.arange(len(m)) for m in ys]
        stds = [
            np.asarray(logger.data[k])
            for k in sorted(logger.data.keys())
            if k.endswith(".std") and k.startswith(query)
        ]
        los = [m - s for m, s in zip(ys, stds)]
        his = [m + s for m, s in zip(ys, stds)]

    return {
        "x": xs,
        "y": ys,
        "lo": [m - s for m, s in zip(means, stds)],
        "hi": [m + s for m, s in zip(means, stds)],
    }

    assert style == "median + quantiles"

    if f"{metric}.quantile_0.5" in logger.data:
        return {
            "x": np.arange(mean.shape[0]),
            "y": logger.data[f"{metric}.quantile_0.5"],
            "lo": logger.data[f"{metric}.quantile_0.25"],
            "hi": logger.data[f"{metric}.quantile_0.75"],
            "min": logger.data[f"{metric}.quantile_0.025"],
            "max": logger.data[f"{metric}.quantile_0.975"],
        }
    return {
        "x": np.arange(mean.shape[0]),
        "y": logger.data[f"{metric}.data"],
    }


def update(_attr, _old, _new):
    data = get_data()
    if not data:
        return

    ci_band.visible = "lo" in data
    range_band.visible = "min" in data

    if "min" in data:
        ymin = min(data["min"])
        ymax = max(data["max"])
    elif "lo" in data:
        ymin = min(data["lo"])
        ymax = max(data["hi"])
    else:
        ymin = min(data["y"])
        ymax = max(data["y"])

    plot.x_range.start = -0.2
    plot.x_range.end = len(data["y"]) - 0.8

    plot.y_range.start = ymin - (ymax - ymin) * 0.1
    plot.y_range.end = ymax + (ymax - ymin) * 0.1

    source.data = data


source = ColumnDataSource(get_data())

plot = figure(
    tools="crosshair,pan,reset,save,wheel_zoom,box_zoom,hover",
    x_range=(0, 1),
    y_range=(0, 1),
)
hover = plot.select({"type": HoverTool})
hover.tooltips = [("epoch", "@x"), ("value", "@y")]
hover.mode = "vline"

# plot.line(x="x", y="y", line_width=3, source=source)
glyph = MultiLine(xs="x", ys="y", line_width=3)
plt.add_glyph(source, glyph)
plot.circle(x="x", y="y", size=6, source=source)
ci_band = Band(
    base="x",
    lower="lo",
    upper="hi",
    source=source,
    fill_color="blue",
    fill_alpha=0.2,
)
range_band = Band(
    base="x",
    lower="min",
    upper="max",
    source=source,
    fill_color="blue",
    fill_alpha=0.2,
)
plot.add_layout(
    Span(location=0, dimension="height", line_color="black", line_width=2)
)
plot.add_layout(
    Span(location=0, dimension="width", line_color="black", line_width=2)
)
plot.add_layout(ci_band)
plot.add_layout(range_band)

metric_input.on_change("value", update)
style_select.on_change("value", update)
update(None, None, None)

curdoc().add_root(column(metric_input, style_select, plot))
