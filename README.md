# EzTuning

A custom-made hyperparameter tuning / experiment framework I made for my own use cases.

Taking inspiration from the
[Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook),
multiple kinds of hyperparameters can be specified. Here, four types are supported:
1. `"task"` parameters can be used to specify different task or environment
configurations. Two trials with different task specifications will have their results
reported independently.
2. `"science"` parameters are used to specify the hyperparameters whose effects we are 
trying to measure. The resulting plots compare the best results between all unique
configurations of science parameters.
3. `"nuisance"` parameters are those which are optimized over in order to fairly compare
different science parameters.
4. `"id"` parameters are used to differentiate independent runs with otherwise
identical parameters. Trials with different IDs but otherwise identical parameters have
their results aggregated together before results are reported.

The resulting plots are stored in a local directory. One plot for each metric displays
the 90% boostrapped confidence intervals of the IQM of that metric over time (following
the guidelines in <https://arxiv.org/abs/2108.13264>). One line
is plotted per unique configuration of science parameters, and only results from the
best nuisance configuration corresponding to each science configuration are used.
Results for each unique task configuration are reported in a separate directory.

Trials are run in parallel using
[ray-core](https://docs.ray.io/en/latest/ray-core/walkthrough.html).
Trials should yield Logger
objects, generator-style. See `tuner_example.py` for example usage.
