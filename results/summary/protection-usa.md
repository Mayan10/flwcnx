# Protection layer: usa

Run `results/protection/usa`, 2,000 decisions, 5 workload seeds, seed 1337. Config snapshot and environment are in `result.json`.

**The capacity is measured and the flows are a model.** No public LEO dataset carries a flow table, process attribution or a user-attention signal, so the capacity series below is the calibrated bound and the realised throughput from a trained pipeline over the real traces, and the flows contending for it come from `thalweg/eval/workload.py`. Every number here is a number about that workload.

| capacity series | |
|---|---|
| decisions | 2,000 |
| calibrated bound, mean | 212.8 Mbps |
| realised throughput, mean | 222.4 Mbps |
| realised risk rate | 0.352 against a budget of 0.35 |
| point forecast risk rate | 0.501 |


## Policies at 2x offered load

| policy | critical violation (allocated) | critical violation (delivered) | critical transfers abandoned | critical goodput | ordinary goodput | link utilisation | ordinary slowdown |
|---|---|---|---|---|---|---|---|
| oracle (perfect classification) | 0.002 +/- 0.002 | 0.030 | 0.000 | 0.952 | 0.918 | 0.618 | 1.09x |
| **criticality-weighted, with floors (ours)** | 0.029 +/- 0.006 | 0.062 | 0.000 | 0.874 | 0.954 | 0.669 | 1.05x |
| class priority (DiffServ) | 0.081 +/- 0.033 | 0.110 | 0.007 | 0.781 | 0.949 | 0.708 | 1.10x |
| equal share (fair queue) | 0.039 +/- 0.021 | 0.067 | 0.000 | 0.869 | 0.955 | 0.671 | 1.04x |
| throttle the largest first | 0.042 +/- 0.018 | 0.067 | 0.000 | 0.860 | 0.959 | 0.678 | 1.03x |

## The load sweep

Under light load nothing has to be shed and every policy looks the same. Under heavy enough load the floors stop fitting and the oracle fails too. Either end on its own would be mistaken for the general case.

| offered load | oracle (perfect classification) | criticality-weighted, with floors (ours) | class priority (DiffServ) | equal share (fair queue) | throttle the largest first |
|---|---|---|---|---|---|
| 1x nominal (0.4x capacity) | 0.000 | 0.006 | 0.007 | 0.005 | 0.005 |
| 1.5x nominal (0.6x capacity) | 0.001 | 0.015 | 0.023 | 0.015 | 0.017 |
| 2x nominal (0.7x capacity) | 0.002 | 0.029 | 0.081 | 0.039 | 0.042 |
| 2.5x nominal (0.9x capacity) | 0.002 | 0.038 | 0.144 | 0.063 | 0.069 |
| 3x nominal (1.1x capacity) | 0.003 | 0.042 | 0.274 | 0.152 | 0.141 |
| 4x nominal (1.6x capacity) | 0.013 | 0.054 | 0.381 | 0.285 | 0.251 |

## Where the violations land

| archetype | must not be shed | mean criticality | violation rate | goodput |
|---|---|---|---|---|
| pacs image push | yes | 0.24 | 0.125 | 0.85 |
| research upload | yes | 0.28 | 0.035 | 0.93 |
| patient monitor | yes | 0.88 | 0.000 | 0.98 |
| teleconsult video | yes | 0.93 | 0.000 | 0.98 |
| video stream | no | 0.42 | 0.005 | 0.96 |
| cloud backup | no | 0.03 | 0.000 | 0.92 |
| greedy downloader | no | 0.55 | 0.000 | 0.95 |
| os update | no | 0.12 | 0.000 | 0.93 |

## Removing a mechanism

| variant | critical violation | change | critical goodput | scorer AP | flap rate |
|---|---|---|---|---|---|
| full | 0.029 +/- 0.006 |  | 0.874 | 0.845 | 0.0016 |
| no floors | 0.040 +/- 0.018 | +0.011 | 0.868 | 0.843 | 0.0018 |
| no hysteresis | 0.030 +/- 0.006 | +0.001 | 0.873 | 0.844 | 0.0032 |
| no priors | 0.026 +/- 0.007 | -0.002 | 0.885 | 0.790 | 0.0015 |
| priors only | 0.043 +/- 0.025 | +0.014 | 0.860 | 0.760 | 0.0000 |
| sharp weights | 0.031 +/- 0.006 | +0.002 | 0.869 | 0.846 | 0.0015 |

## Removing an evidence channel

A channel whose removal improves the outcome is not carrying its weight. The defaults were not retuned on the strength of this table, because fitting the weights on the evaluation workload and then evaluating on the same generator is the circularity the module docstring warns about.

| channel removed | critical violation | change | critical goodput | scorer AP | flap rate |
|---|---|---|---|---|---|
| attention | 0.028 +/- 0.005 | -0.001 | 0.871 | 0.849 | 0.0003 |
| deadline | 0.022 +/- 0.009 | -0.007 | 0.893 | 0.903 | 0.0012 |
| elasticity | 0.040 +/- 0.010 | +0.011 | 0.859 | 0.831 | 0.0004 |
| interactivity | 0.021 +/- 0.004 | -0.007 | 0.880 | 0.799 | 0.0029 |
| irreversibility | 0.035 +/- 0.007 | +0.006 | 0.862 | 0.829 | 0.0011 |
| none | 0.029 +/- 0.006 |  | 0.874 | 0.845 | 0.0016 |
| recurrence | 0.035 +/- 0.007 | +0.006 | 0.863 | 0.827 | 0.0018 |
| volume | 0.027 +/- 0.005 | -0.002 | 0.875 | 0.850 | 0.0013 |

## Dividing the bound against dividing the forecast

The allocator honours every feasible floor in both arms. Against the point forecast it writes *more* floors, because the number is larger, and the link then fails them.

| capacity divided | violation (allocated) | violation (delivered) |
|---|---|---|
| bound | 0.029 | 0.062 |
| point forecast | 0.024 | 0.064 |

## The weights are hand set, so they were perturbed

24 draws, each channel weight multiplied by a log-uniform factor spanning a factor of four either way. Critical violation ranged 0.003 to 0.044, median 0.020, against 0.029 at the hand-set values. The best deployable baseline at this load is 0.024, and 15 of 24 draws beat it.


## The workload

| archetype | declared | must not be shed | nominal | floor | elastic | deadline |
|---|---|---|---|---|---|---|
| patient monitor | life_safety | yes | 0.5 Mbps | 0.4 Mbps | no | none |
| pacs image push | standard | yes | 28 Mbps | 15.4 Mbps | no | 1.6x |
| teleconsult video | clinical | yes | 3 Mbps | 1.65 Mbps | no | none |
| research upload | standard | yes | 16 Mbps | 7.2 Mbps | no | 1.35x |
| video stream | standard | no | 9 Mbps | 2.7 Mbps | no | none |
| greedy downloader | clinical | no | 45 Mbps | 0 Mbps | yes | none |
| os update | standard | no | 32 Mbps | 0 Mbps | yes | none |
| cloud backup | background | no | 22 Mbps | 0 Mbps | yes | 12x |
