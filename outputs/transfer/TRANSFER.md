# Cross-scene transfer (train on one scene, test on another; no retraining)

Common features: 400

| model | norm | train | test | head | n | positives | base rate | AUROC | 95% CI | AP | within-scene AUROC | within-scene AP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| logreg | train | HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.786 | [0.77, 0.81] | 0.498 | 0.864 | 0.702 |
| logreg | train | HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.814 | [0.80, 0.83] | 0.964 | 0.929 | 0.988 |
| logreg | train | HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.794 | [0.73, 0.84] | 0.019 | 0.906 | 0.102 |
| logreg | train | AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.840 | [0.78, 0.89] | 0.569 | 0.934 | 0.755 |
| logreg | train | AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.907 | [0.86, 0.93] | 0.984 | 0.968 | 0.995 |
| logreg | train | AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.772 | [0.69, 0.85] | 0.005 | 0.914 | 0.051 |
| logreg | own_rank | HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.753 | [0.71, 0.79] | 0.435 | 0.864 | 0.702 |
| logreg | own_rank | HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.829 | [0.81, 0.85] | 0.968 | 0.929 | 0.988 |
| logreg | own_rank | HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.830 | [0.78, 0.87] | 0.011 | 0.906 | 0.102 |
| logreg | own_rank | AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.793 | [0.74, 0.85] | 0.516 | 0.934 | 0.755 |
| logreg | own_rank | AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.922 | [0.88, 0.95] | 0.987 | 0.968 | 0.995 |
| logreg | own_rank | AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.875 | [0.82, 0.90] | 0.010 | 0.914 | 0.051 |
| gbdt | train | HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.696 | [0.65, 0.74] | 0.377 | 0.920 | 0.796 |
| gbdt | train | HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.805 | [0.79, 0.82] | 0.963 | 0.950 | 0.991 |
| gbdt | train | HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.777 | [0.69, 0.83] | 0.009 | 0.894 | 0.038 |
| gbdt | train | AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.900 | [0.86, 0.93] | 0.672 | 0.941 | 0.764 |
| gbdt | train | AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.931 | [0.89, 0.96] | 0.988 | 0.976 | 0.996 |
| gbdt | train | AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.940 | [0.92, 0.95] | 0.067 | 0.929 | 0.260 |
| gbdt | own_rank | HKairport_GNSS03 | AMtown03 | failure | 30477 | 6749 | 0.221 | 0.696 | [0.66, 0.74] | 0.356 | 0.920 | 0.796 |
| gbdt | own_rank | HKairport_GNSS03 | AMtown03 | layered | 147110 | 128919 | 0.876 | 0.822 | [0.80, 0.84] | 0.966 | 0.950 | 0.991 |
| gbdt | own_rank | HKairport_GNSS03 | AMtown03 | vehicle | 357783 | 608 | 0.002 | 0.766 | [0.70, 0.82] | 0.008 | 0.894 | 0.038 |
| gbdt | own_rank | AMtown03 | HKairport_GNSS03 | failure | 10095 | 2206 | 0.219 | 0.900 | [0.86, 0.93] | 0.657 | 0.941 | 0.764 |
| gbdt | own_rank | AMtown03 | HKairport_GNSS03 | layered | 48239 | 42950 | 0.890 | 0.920 | [0.88, 0.95] | 0.986 | 0.976 | 0.996 |
| gbdt | own_rank | AMtown03 | HKairport_GNSS03 | vehicle | 105236 | 221 | 0.002 | 0.923 | [0.88, 0.95] | 0.059 | 0.929 | 0.260 |