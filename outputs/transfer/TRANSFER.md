# Cross-scene transfer (train on one scene, test on another; no retraining)

Common features: 400

| model | norm | train | test | head | n | positives | base rate | AUROC | 95% CI | AP | within-scene AUROC | within-scene AP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | train | HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.785 | [0.76, 0.81] | 0.495 | 0.864 | 0.702 |
| logreg | train | HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.814 | [0.80, 0.83] | 0.964 | 0.929 | 0.988 |
| logreg | train | HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.791 | [0.73, 0.83] | 0.015 | 0.906 | 0.102 |
| logreg | train | AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.840 | [0.78, 0.89] | 0.569 | 0.933 | 0.755 |
| logreg | train | AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.907 | [0.86, 0.93] | 0.984 | 0.968 | 0.995 |
| logreg | train | AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.771 | [0.70, 0.85] | 0.006 | 0.909 | 0.047 |
| logreg | own_rank | HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.749 | [0.71, 0.78] | 0.428 | 0.864 | 0.702 |
| logreg | own_rank | HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.828 | [0.81, 0.85] | 0.968 | 0.929 | 0.988 |
| logreg | own_rank | HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.830 | [0.78, 0.87] | 0.011 | 0.906 | 0.102 |
| logreg | own_rank | AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.793 | [0.74, 0.85] | 0.517 | 0.933 | 0.755 |
| logreg | own_rank | AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.922 | [0.88, 0.95] | 0.987 | 0.968 | 0.995 |
| logreg | own_rank | AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.876 | [0.83, 0.90] | 0.010 | 0.909 | 0.047 |
| gbdt | train | HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.691 | [0.64, 0.74] | 0.374 | 0.920 | 0.796 |
| gbdt | train | HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.814 | [0.80, 0.83] | 0.965 | 0.950 | 0.991 |
| gbdt | train | HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.784 | [0.71, 0.84] | 0.010 | 0.894 | 0.038 |
| gbdt | train | AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.899 | [0.86, 0.93] | 0.672 | 0.940 | 0.762 |
| gbdt | train | AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.931 | [0.89, 0.96] | 0.988 | 0.976 | 0.996 |
| gbdt | train | AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.939 | [0.92, 0.95] | 0.066 | 0.935 | 0.297 |
| gbdt | own_rank | HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.689 | [0.64, 0.74] | 0.360 | 0.920 | 0.796 |
| gbdt | own_rank | HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.821 | [0.80, 0.84] | 0.966 | 0.950 | 0.991 |
| gbdt | own_rank | HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.765 | [0.69, 0.82] | 0.008 | 0.894 | 0.038 |
| gbdt | own_rank | AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.900 | [0.86, 0.93] | 0.657 | 0.940 | 0.762 |
| gbdt | own_rank | AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.920 | [0.88, 0.95] | 0.986 | 0.976 | 0.996 |
| gbdt | own_rank | AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.923 | [0.87, 0.95] | 0.060 | 0.935 | 0.297 |