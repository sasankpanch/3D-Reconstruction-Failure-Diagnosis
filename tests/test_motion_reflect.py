import numpy as np, pandas as pd
from aerodiag.grid.ground_grid import GroundGrid
from aerodiag.diagnosis.motion_reflect import fit_motion_reflectance

def test_moving_vs_reflective_separation_and_pooling():
    rng = np.random.default_rng(0); g = GroundGrid(0, 0, 1, 10, 10); rows = []
    for cell, kind, nobs in [(11, "moving", 40), (55, "reflective", 40), (77, "static", 40), (88, "static", 3)]:
        dt = rng.uniform(0.1, 60, nobs); dth = rng.uniform(0, 40, nobs)
        across = 0.2 + rng.normal(0, 0.1, nobs) + (0.5 * dt if kind == "moving" else 0)
        pho = 0.1 + rng.normal(0, 0.02, nobs) + (0.01 * dth if kind == "reflective" else 0)
        rows.append(pd.DataFrame(dict(cell_id=cell, dt_s=dt, dtheta_deg=dth, across_px=across, cost_min=pho, gain=rng.normal(0, 0.02, nobs))))
    rows.append(pd.DataFrame(dict(cell_id=87, dt_s=rng.uniform(0.1, 60, 20), dtheta_deg=rng.uniform(0, 40, 20),
                                  across_px=0.2 + rng.normal(0, 0.1, 20), cost_min=0.1 + rng.normal(0, 0.02, 20), gain=rng.normal(0, 0.02, 20))))
    F = fit_motion_reflectance(pd.concat(rows), g, min_obs=6)
    assert F.moving_score[11] > 10 and F.moving_score[77] < 3 and F.moving_score[55] < 3
    assert F.reflective_score[55] > 10 and F.reflective_score[11] < 3
    assert F.pool_radius[88] == 1 and F.n_obs_fit[88] >= 6 and F.pool_radius[77] == 0
