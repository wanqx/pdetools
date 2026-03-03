# pdetools

## Frozen Optical CJ Pipeline

This repository freezes the current optical-only path:

`Areal_2light (ch0/ch1) -> TOF velocity -> M_D -> optical-estimated p_CJ`

Validation uses `Areal_4highf_p` plateau only, and does not feed back into optical estimation.

## Module

`optical_cj_pipeline.py`

## Public Interfaces

### `get_frozen_build_options()`
Return the frozen default options used for detonation-cycle construction from optical channels.

### `estimate_optical_cj_from_cycle_result(cycle_result)`
Estimate cycle-wise optical CJ pressure from a prepared `cycle_result`.

Parameters:
- `cycle_result` (`dict`): output from `build_detonation_cycle_dataset`.

Returns (`dict`):
- `cycle_time`
- `velocity_raw`
- `velocity`
- `velocity_for_cj`
- `M_D`
- `p_cj_optical`
- `p_cj_measured`
- `peak_p_a4`
- `validation_mae_pa`
- `validation_mape`
- plus quality/diagnostic arrays (`confidence`, `quality_low`, `expected_blend`, etc.)

### `run_optical_to_cj_pipeline(Areal_1time, Areal_2light, areal_data=None, build_options=None)`
Run the full frozen pipeline end-to-end.

Parameters:
- `Areal_1time` (`array-like`): time axis (seconds)
- `Areal_2light` (`array-like`): optical channels; only columns 0 and 1 are used
- `areal_data` (`dict`, optional): `Areal_*` arrays packaged into slices
- `build_options` (`dict`, optional): overrides on top of frozen build defaults

Returns (`dict`):
- `cycle_result`
- `cj_result`

## Jupyter Usage

```python
from optical_cj_pipeline import run_optical_to_cj_pipeline
import numpy as np

areal_data = {
    k: np.asarray(v)
    for k, v in data.items()
    if k.startswith("Areal_") and np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] == len(Areal_1time)
}

pipeline = run_optical_to_cj_pipeline(Areal_1time, Areal_2light, areal_data=areal_data)
cycle_result = pipeline["cycle_result"]
cj_result = pipeline["cj_result"]
print(cj_result["validation_mae_pa"], cj_result["validation_mape"])
```
