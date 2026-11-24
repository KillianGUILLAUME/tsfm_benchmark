### Daily log realized variance, horizon 1 day  (n = 8304 forecasts)

| model | family | pinball | skill vs har_dow | DM p | cov80 | calib | MAE(median) | sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| har_dow | classical | 0.2270 | ref |  | 0.801 | 0.029 | 0.582 | 0 |
| chronos_bolt_small | foundation zero-shot | 0.2314 | -1.9% | 0.013 | 0.758 | 0.023 | 0.592 | 79 |
| chronos_bolt_tiny | foundation zero-shot | 0.2392 | -5.4% | 0.000 | 0.760 | 0.022 | 0.612 | 16 |
| har | classical | 0.2463 | -8.5% | 0.000 | 0.804 | 0.016 | 0.629 | 0 |
| arima | classical | 0.2502 | -10.2% | 0.000 | 0.806 | 0.026 | 0.641 | 248 |
| local_level | classical | 0.2531 | -11.5% | 0.000 | 0.807 | 0.006 | 0.648 | 48 |
| lstm_attn | trained from scratch | 0.2577 | -13.5% | 0.000 | 0.790 | 0.009 | 0.652 | 0 |
| ewma | classical | 0.2635 | -16.0% | 0.000 | 0.808 | 0.012 | 0.672 | 0 |
| naive | classical | 0.2929 | -29.0% | 0.000 | 0.793 | 0.004 | 0.750 | 0 |

### Daily log realized variance, horizon 5 days  (n = 8304 forecasts)

| model | family | pinball | skill vs har_dow | DM p | cov80 | calib | MAE(median) | sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| har_dow | classical | 0.2581 | ref |  | 0.812 | 0.047 | 0.663 | 0 |
| chronos_bolt_small | foundation zero-shot | 0.2607 | -1.0% | 0.158 | 0.791 | 0.021 | 0.666 | 79 |
| chronos_bolt_tiny | foundation zero-shot | 0.2680 | -3.8% | 0.000 | 0.794 | 0.016 | 0.685 | 16 |
| har | classical | 0.2741 | -6.2% | 0.000 | 0.808 | 0.037 | 0.704 | 0 |
| lstm_attn | trained from scratch | 0.2773 | -7.4% | 0.000 | 0.791 | 0.007 | 0.702 | 0 |
| ewma | classical | 0.2780 | -7.7% | 0.000 | 0.808 | 0.015 | 0.710 | 0 |
| arima | classical | 0.2816 | -9.1% | 0.000 | 0.808 | 0.059 | 0.726 | 248 |
| local_level | classical | 0.2890 | -12.0% | 0.000 | 0.811 | 0.010 | 0.742 | 48 |
| naive | classical | 0.3562 | -38.0% | 0.000 | 0.812 | 0.007 | 0.914 | 0 |

### Daily log-return, horizon 1 day  (n = 8304 forecasts)

| model | family | pinball x1000 | skill vs garch_t | DM p | cov80 | calib | R2_oos(median) | sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| garch_t | classical | 10.3940 | ref |  | 0.820 | 0.017 | -0.01% | 51 |
| chronos_bolt_small | foundation zero-shot | 10.4320 | -0.4% | 0.082 | 0.771 | 0.011 | -0.75% | 347 |
| hist_sim | classical | 10.4371 | -0.4% | 0.050 | 0.815 | 0.010 | -0.45% | 0 |
| chronos_bolt_tiny | foundation zero-shot | 10.4760 | -0.8% | 0.014 | 0.753 | 0.019 | -1.27% | 17 |
| ewma_vol | classical | 10.4829 | -0.9% | 0.000 | 0.836 | 0.033 | +0.00% | 0 |
| lstm_attn | trained from scratch | 10.5166 | -1.2% | 0.005 | 0.779 | 0.006 | -1.46% | 0 |
