# Onboard benchmark — HKairport_GNSS03

Hardware: {'gpu': 'NVIDIA GeForce RTX 5080', 'cpu': 'x86_64'}

| step | CPU | GPU |
|---|---|---|
| decode_s_per_image | 0.0079 | (CPU) |
| sift_cpu_1thread_s_per_image | 0.6005 | (CPU) |
| matching_s_per_pair | 0.1125 | 0.0082 |
| refine_s_per_pair | 0.0027 | 0.0040 |
| sweep_s_per_pair | 5.4481 | 0.1651 |
| sweep_cells_per_pair | 21599.0000 | 21599.0000 |
| sweep_s_per_1k_cells | 0.2500 | 0.0074 |
| accumulate_s_per_1M_rows | 4.2431 | 0.0923 |
| inference_s_per_10k_cells | 0.0931 | 0.0465 |
| per_image_dataset_spacing_s | 12.5278 | 0.4723 |
| pairs_per_image_dataset | 2.2254 | 2.2254 |
| per_image_survey_spacing_s | 27.9636 | 0.9642 |
| sift_runs_in_parallel_on_cpu_s | 0.6005 | 0.6005 |