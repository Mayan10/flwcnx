# Phase 1 gate: StarNet reproduction

Run 2026-09-01 on the released traces. **The gate passes.** Look-back 30,
output 5, their published per-location step sizes (46 / 6 / 29), contiguous 8:2
split, 50 epochs with early stopping, seed 1337, Apple MPS.

## Result

| location | RMSE | published | gap | MAE | published | gap | median abs err |
|---|---|---|---|---|---|---|---|
| USA (CHI) | 42.56 | 40.33 | +5.5% | 31.94 | 29.88 | +6.9% | 24.69 |
| Canada (VIC) | 38.38 | 41.08 | **-6.6%** | 29.55 | 30.84 | **-4.2%** | 23.49 |
| Germany (OSN) | 38.24 | 36.48 | +4.8% | 28.42 | 27.11 | +4.8% | 21.41 |
| **average** | **39.73** | **39.30** | **+1.1%** | **29.97** | **29.28** | **+2.4%** |

The average lands within 1.1% RMSE and 2.4% MAE. Per-location gaps are 4.2% to
6.9% and they scatter in **both** directions, with Canada coming out better
than published and the other two worse. That pattern is what an independent
reimplementation looks like. A reimplementation that had been tuned toward the
target would sit just under it everywhere.

Two known reasons the per-location numbers should not land exactly:

1. **We do not reproduce their scaler leak.** Their `get_data_loader` fits the
   scaler on the whole trace before splitting (`docs/data.md`). Ours fits on
   training windows only. Theirs is the easier problem.
2. **Hidden size.** The paper text says 128 and their code defaults to 60. We
   used 128, following the text.

## The loader verifies exactly

| location | samples | published | satellites | published |
|---|---|---|---|---|
| USA | 1,123,832 | 1,123,832 | 5,723 | 5,723 |
| Canada | 145,053 | 145,053 | 3,166 | 3,166 |
| Germany | 613,295 | 613,295 | 3,956 | 3,956 |

Canada and Germany match the paper's dataset table exactly. The US file needed
explaining, because the paper reports collecting 2,475,163 samples and the
released file holds 1,123,832. It is not a truncated download:

- 1,123,832 is exactly the CHI sample count BG-CFQS report processing;
- `(1,123,832 - 45) / 46 + 1 = 24,430`, exactly the US data-point count StarNet
  report training on at sequence length 45 and step 46.

The released file is their training set; the collection figure counts raw
measurement minutes that never reached it. `replay.py` now checks against
`RELEASED_FILE_STATS` and carries the collection figures alongside without
asserting on them.

## One number that does not compare, and is not claimed

Our median absolute error averages 23.20 Mbps against their reported 33.57.
**This is not a 31% improvement and must not be reported as one.** Their 33.57
appears in the context of the T3P comparison (33.57 against 43.63, "30.3%
better"), which is their main configuration at output length 15, not the
output-5 setting this table uses. Until it is confirmed which horizon that
figure is measured at, the two are not the same quantity. The RMSE and MAE
comparison above is like-for-like and is the one the gate rests on.

## Not yet run

The ablations (published: 38.00 without the periodical embedding, 37.01 without
attention). `scripts/reproduce_starnet.py --ablations`.
