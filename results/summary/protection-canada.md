# Protection layer: canada

Run `results/protection/canada`, 2,000 decisions, 5 workload seeds, seed 1337. Config snapshot and environment are in `result.json`.

**The capacity is measured and the flows are a model.** No public LEO dataset carries a flow table, process attribution or a user-attention signal, so the capacity series below is the calibrated bound and the realised throughput from a trained pipeline over the real traces, and the flows contending for it come from `thalweg/eval/workload.py`. Every number here is a number about that workload.

| capacity series | |
|---|---|
| decisions | 2,000 |
| calibrated bound, mean | 136.2 Mbps |
| realised throughput, mean | 145.9 Mbps |
| realised risk rate | 0.351 against a budget of 0.35 |
| point forecast risk rate | 0.490 |


## Policies at 2x offered load

| policy | critical violation (allocated) | critical violation (delivered) | critical transfers abandoned | critical goodput | ordinary goodput | link utilisation | ordinary slowdown |
|---|---|---|---|---|---|---|---|
| oracle (perfect classification) | 0.029 +/- 0.014 | 0.109 | 0.000 | 0.746 | 0.651 | 0.731 | 1.39x |
| **criticality-weighted, with floors (ours)** | 0.065 +/- 0.014 | 0.203 | 0.003 | 0.581 | 0.687 | 0.788 | 1.46x |
| class priority (DiffServ) | 0.301 +/- 0.048 | 0.348 | 0.232 | 0.431 | 0.802 | 0.822 | 1.39x |
| equal share (fair queue) | 0.251 +/- 0.049 | 0.298 | 0.130 | 0.508 | 0.836 | 0.791 | 1.22x |
| throttle the largest first | 0.218 +/- 0.038 | 0.258 | 0.121 | 0.474 | 0.910 | 0.794 | 1.09x |

## The load sweep

Under light load nothing has to be shed and every policy looks the same. Under heavy enough load the floors stop fitting and the oracle fails too. Either end on its own would be mistaken for the general case.

| offered load | oracle (perfect classification) | criticality-weighted, with floors (ours) | class priority (DiffServ) | equal share (fair queue) | throttle the largest first |
|---|---|---|---|---|---|
| 1x nominal (0.6x capacity) | 0.001 | 0.017 | 0.030 | 0.022 | 0.024 |
| 1.5x nominal (0.9x capacity) | 0.008 | 0.039 | 0.160 | 0.117 | 0.110 |
| 2x nominal (1.3x capacity) | 0.029 | 0.065 | 0.301 | 0.251 | 0.218 |
| 2.5x nominal (1.6x capacity) | 0.046 | 0.088 | 0.393 | 0.307 | 0.276 |
| 3x nominal (2.0x capacity) | 0.088 | 0.122 | 0.412 | 0.373 | 0.341 |
| 4x nominal (2.7x capacity) | 0.175 | 0.222 | 0.384 | 0.408 | 0.373 |

## Where the violations land

| archetype | must not be shed | mean criticality | violation rate | goodput |
|---|---|---|---|---|
| pacs image push | yes | 0.72 | 0.217 | 0.52 |
| research upload | yes | 0.71 | 0.059 | 0.66 |
| patient monitor | yes | 0.89 | 0.001 | 0.94 |
| teleconsult video | yes | 0.94 | 0.000 | 0.89 |
| video stream | no | 0.57 | 0.198 | 0.62 |
| cloud backup | no | 0.03 | 0.000 | 0.86 |
| greedy downloader | no | 0.52 | 0.000 | 0.86 |
| os update | no | 0.11 | 0.000 | 0.81 |

## Removing a mechanism

| variant | critical violation | change | critical goodput | scorer AP | flap rate |
|---|---|---|---|---|---|
| full | 0.065 +/- 0.014 |  | 0.581 | 0.948 | 0.0069 |
| no floors | 0.229 +/- 0.045 | +0.164 | 0.536 | 0.947 | 0.0047 |
| no hysteresis | 0.067 +/- 0.014 | +0.002 | 0.578 | 0.948 | 0.0226 |
| no priors | 0.086 +/- 0.022 | +0.021 | 0.590 | 0.863 | 0.0069 |
| priors only | 0.257 +/- 0.046 | +0.192 | 0.498 | 0.765 | 0.0000 |
| sharp weights | 0.064 +/- 0.015 | -0.001 | 0.587 | 0.947 | 0.0067 |

## Removing an evidence channel

A channel whose removal improves the outcome is not carrying its weight. The defaults were not retuned on the strength of this table, because fitting the weights on the evaluation workload and then evaluating on the same generator is the circularity the module docstring warns about.

| channel removed | critical violation | change | critical goodput | scorer AP | flap rate |
|---|---|---|---|---|---|
| attention | 0.061 +/- 0.015 | -0.004 | 0.601 | 0.955 | 0.0027 |
| deadline | 0.123 +/- 0.026 | +0.058 | 0.561 | 0.919 | 0.0037 |
| elasticity | 0.100 +/- 0.019 | +0.034 | 0.570 | 0.935 | 0.0021 |
| interactivity | 0.061 +/- 0.017 | -0.004 | 0.582 | 0.911 | 0.0034 |
| irreversibility | 0.073 +/- 0.017 | +0.008 | 0.565 | 0.939 | 0.0075 |
| none | 0.065 +/- 0.014 |  | 0.581 | 0.948 | 0.0069 |
| recurrence | 0.077 +/- 0.015 | +0.011 | 0.564 | 0.936 | 0.0059 |
| volume | 0.061 +/- 0.015 | -0.004 | 0.582 | 0.950 | 0.0048 |

## Dividing the bound against dividing the forecast

The allocator honours every feasible floor in both arms. Against the point forecast it writes *more* floors, because the number is larger, and the link then fails them.

| capacity divided | violation (allocated) | violation (delivered) |
|---|---|---|
| bound | 0.065 | 0.203 |
| point forecast | 0.049 | 0.244 |

## The weights are hand set, so they were perturbed

24 draws, each channel weight multiplied by a log-uniform factor spanning a factor of four either way. Critical violation ranged 0.048 to 0.144, median 0.079, against 0.065 at the hand-set values. The best deployable baseline at this load is 0.150, and 24 of 24 draws beat it.


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
