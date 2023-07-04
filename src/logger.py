import math
from collections import defaultdict

import numpy as np
from ddsketch import DDSketch


class RunningStats:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.s = 0

        self.quantile_estimator = DDSketch()

    def extend(self, arr):
        for x in arr:
            self.push(float(x))

    def push(self, x):
        assert isinstance(x, (float, int))
        self.n += 1
        if self.n == 1:
            self.m = x
        else:
            old_m = self.m
            self.m = old_m + (x - old_m) / self.n
            self.s = self.s + (x - old_m) * (x - self.m)

        self.quantile_estimator.add(x)

    def mean(self):
        return self.m

    def std(self):
        if self.n <= 1:
            return self.m
        return math.sqrt(self.s / (self.n - 1))

    def quantile(self, q: float):
        assert 0 <= q <= 1
        return self.quantile_estimator.get_quantile_value(q)


DEFAULT_QUANTILES = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]


class Logger:
    def __init__(self, quantiles=None):
        self.buffer = defaultdict(RunningStats)

        self.data = defaultdict(list)
        self.quantiles = DEFAULT_QUANTILES if quantiles is None else quantiles

    # log metrics reported once per epoch
    def log(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "shape"):
                v = v.item()
            self.data[k + ".data"].append(v)

    # push metrics logged many times per epoch, to aggregate later
    def push(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "shape"):
                v = v.item()
            self.buffer[k].push(v)

    # pushes a batch of metrics logged many times per epoch
    def extend(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in {**metrics, **kwargs}.items():
            if hasattr(v, "grad"):
                v = v.detach().numpy()
            assert len(v.shape) == 1
            self.buffer[k].extend(v)

    # compute and log aggregate statistics for pushed metrics
    def step(self):
        for k, v in self.buffer.items():
            self.data[f"{k}.data"].append(v.mean())
            self.data[f"{k}.std"].append(v.std())
            for q in self.quantiles:
                self.data[f"{k}.quantile_{q}"].append(v.quantile(q))
        self.buffer.clear()

    def save(self, filename):
        with open(filename, "wb") as f:
            np.savez_compressed(
                f,
                **{
                    k: np.asarray(v, dtype=np.float32)
                    for k, v in self.data.items()
                },
            )

    @staticmethod
    def load(filename):
        data = np.load(filename)
        ret = Logger()
        for k, v in data.items():
            ret.data[k] = v
        return ret
