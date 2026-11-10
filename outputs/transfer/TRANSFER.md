# Cross-scene transfer (train on one scene, test on another; no retraining)

Common features: 78

| train | test | head | n | positives | base rate | AUROC | 95% CI | AP | within-scene AUROC | within-scene AP |
|---|---|---|---|---|---|---|---|---|---|---|
| HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.726 | [0.69, 0.76] | 0.421 | 0.860 | 0.651 |
| HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.776 | [0.76, 0.80] | 0.956 | 0.910 | 0.984 |
| HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.785 | [0.73, 0.83] | 0.010 | 0.874 | 0.027 |
| AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.790 | [0.73, 0.85] | 0.433 | 0.926 | 0.720 |
| AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.855 | [0.81, 0.88] | 0.977 | 0.947 | 0.992 |
| AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.791 | [0.68, 0.85] | 0.027 | 0.938 | 0.055 |