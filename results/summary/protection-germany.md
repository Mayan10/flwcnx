# Protection layer: germany

Run `results/protection/germany`, 2,000 decisions, 5 workload seeds, seed 1337. Config snapshot and environment are in `result.json`.

**The capacity is measured and the flows are a model.** No public LEO dataset carries a flow table, process attribution or a user-attention signal, so the capacity series below is the calibrated bound and the realised throughput from a trained pipeline over the real traces, and the flows contending for it come from `flwcnx/eval/workload.py`. Every number here is a number about that workload.

| capacity series | |
|---|---|
| decisions | 2,000 |
| calibrated bound, mean | 143.0 Mbps |
| realised throughput, mean | 152.6 Mbps |
| realised risk rate | 0.348 against a budget of 0.35 |
| point forecast risk rate | 0.586 |


## Policies at 2x offered load

| policy | critical violation (allocated) | critical violation (delivered) | critical transfers abandoned | critical goodput | ordinary goodput | link utilisation | ordinary slowdown |
|---|---|---|---|---|---|---|---|
| oracle (perfect classification) | 0.012 +/- 0.006 | 0.076 | 0.000 | 0.838 | 0.658 | 0.733 | 1.58x |
| **criticality-weighted, with floors (ours)** | 0.048 +/- 0.007 | 0.142 | 0.000 | 0.679 | 0.795 | 0.769 | 1.26x |
| class priority (DiffServ) | 0.271 +/- 0.063 | 0.317 | 0.104 | 0.485 | 0.839 | 0.822 | 1.28x |
| equal share (fair queue) | 0.185 +/- 0.048 | 0.231 | 0.053 | 0.600 | 0.866 | 0.778 | 1.15x |
| throttle the largest first | 0.165 +/- 0.043 | 0.206 | 0.038 | 0.580 | 0.908 | 0.785 | 1.08x |

## The load sweep

Under light load nothing has to be shed and every policy looks the same. Under heavy enough load the floors stop fitting and the oracle fails too. Either end on its own would be mistaken for the general case.

| offered load | oracle (perfect classification) | criticality-weighted, with floors (ours) | class priority (DiffServ) | equal share (fair queue) | throttle the largest first |
|---|---|---|---|---|---|
| 1x nominal (0.5x capacity) | 0.002 | 0.016 | 0.023 | 0.018 | 0.019 |
| 1.5x nominal (0.8x capacity) | 0.006 | 0.039 | 0.093 | 0.061 | 0.062 |
| 2x nominal (1.1x capacity) | 0.012 | 0.048 | 0.271 | 0.185 | 0.165 |
| 2.5x nominal (1.5x capacity) | 0.017 | 0.056 | 0.381 | 0.290 | 0.248 |
| 3x nominal (1.9x capacity) | 0.054 | 0.099 | 0.410 | 0.375 | 0.313 |
| 4x nominal (2.6x capacity) | 0.153 | 0.200 | 0.396 | 0.416 | 0.364 |

## Where the violations land

| archetype | must not be shed | mean criticality | violation rate | goodput |
|---|---|---|---|---|
| pacs image push | yes | 0.63 | 0.177 | 0.59 |
| research upload | yes | 0.64 | 0.040 | 0.72 |
| patient monitor | yes | 0.89 | 0.001 | 0.95 |
| teleconsult video | yes | 0.94 | 0.000 | 0.92 |
| video stream | no | 0.53 | 0.113 | 0.74 |
| cloud backup | no | 0.03 | 0.000 | 0.72 |
| greedy downloader | no | 0.52 | 0.000 | 0.81 |
| os update | no | 0.12 | 0.000 | 0.78 |

## Removing a mechanism

| variant | critical violation | change | critical goodput | scorer AP | flap rate |
|---|---|---|---|---|---|
| full | 0.048 +/- 0.007 |  | 0.679 | 0.928 | 0.0055 |
| no floors | 0.162 +/- 0.044 | +0.114 | 0.632 | 0.930 | 0.0046 |
| no hysteresis | 0.051 +/- 0.007 | +0.003 | 0.677 | 0.927 | 0.0157 |
| no priors | 0.059 +/- 0.012 | +0.010 | 0.688 | 0.847 | 0.0056 |
| priors only | 0.191 +/- 0.048 | +0.143 | 0.589 | 0.774 | 0.0000 |
| sharp weights | 0.050 +/- 0.009 | +0.002 | 0.683 | 0.926 | 0.0055 |

## Removing an evidence channel

A channel whose removal improves the outcome is not carrying its weight. The defaults were not retuned on the strength of this table, because fitting the weights on the evaluation workload and then evaluating on the same generator is the circularity the module docstring warns about.

| channel removed | critical violation | change | critical goodput | scorer AP | flap rate |
|---|---|---|---|---|---|
| attention | 0.046 +/- 0.008 | -0.002 | 0.683 | 0.937 | 0.0025 |
| deadline | 0.081 +/- 0.021 | +0.033 | 0.669 | 0.932 | 0.0031 |
| elasticity | 0.084 +/- 0.013 | +0.036 | 0.654 | 0.909 | 0.0019 |
| interactivity | 0.038 +/- 0.008 | -0.010 | 0.684 | 0.894 | 0.0032 |
| irreversibility | 0.057 +/- 0.009 | +0.009 | 0.666 | 0.913 | 0.0053 |
| none | 0.048 +/- 0.007 |  | 0.679 | 0.928 | 0.0055 |
| recurrence | 0.062 +/- 0.009 | +0.014 | 0.665 | 0.912 | 0.0049 |
| volume | 0.042 +/- 0.008 | -0.006 | 0.682 | 0.932 | 0.0049 |

## Dividing the bound against dividing the forecast

The allocator honours every feasible floor in both arms. Against the point forecast it writes *more* floors, because the number is larger, and the link then fails them.

| capacity divided | violation (allocated) | violation (delivered) |
|---|---|---|
| bound | 0.048 | 0.142 |
| point forecast | 0.041 | 0.171 |

## The weights are hand set, so they were perturbed

24 draws, each channel weight multiplied by a log-uniform factor spanning a factor of four either way. Critical violation ranged 0.027 to 0.122, median 0.051, against 0.048 at the hand-set values. The best deployable baseline at this load is 0.097, and 22 of 24 draws beat it.


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
