# Stage-6 Exact p=0 mathematics suite report

## Starting point from Stage 4 and Stage 5

```json
{
  "stage4_counts_exact_p0": {
    "strict": 45,
    "drift_parrondo": 21,
    "pp_primary": 0,
    "pp_sensitivity": 1,
    "mismatch_sign": 1,
    "mismatch_sign_denominator": 50
  },
  "stage5_T300_fixed": {
    "policy": "fixed",
    "T": 300,
    "T0": 200,
    "T1": 250,
    "strict_raw": 72,
    "drift_raw": 43,
    "strict_stage3": 45,
    "drift_stage3": 21,
    "pp_primary": 0,
    "pp_sensitivity": 1,
    "mismatch_sign": 1,
    "mismatch_sign_denominator": 50,
    "max_adv_phi": 1.0471975511965976,
    "max_adv_phi_over_pi": 0.3333333333333333,
    "max_adv_v": 0.15918236631390129,
    "max_adv_w_loc3": 0.5539169355050063,
    "max_adv_w_loc5": 0.5606677055645829,
    "max_adv_v_fit_ABB": 0.12039081438978931
  },
  "stage5_Tlast_fixed": {
    "policy": "fixed",
    "T": 1500,
    "T0": 200,
    "T1": 250,
    "strict_raw": 72,
    "drift_raw": 43,
    "strict_stage3": 47,
    "drift_stage3": 22,
    "pp_primary": 1,
    "pp_sensitivity": 1,
    "mismatch_sign": 1,
    "mismatch_sign_denominator": 52,
    "max_adv_phi": 0.9599310885968813,
    "max_adv_phi_over_pi": 0.3055555555555556,
    "max_adv_v": 0.15928135925482548,
    "max_adv_w_loc3": 0.5733079652870073,
    "max_adv_w_loc5": 0.5796244638749471,
    "max_adv_v_fit_ABB": 0.11154398900095333
  }
}
```

## Region refinement outputs

- summary json: `region_refine_summary.json`
- full refined grid: `region_refine_grid.csv`
- sampled mask intervals: `region_refine_intervals.csv`

## Long exact timeseries outputs

- summary json: `exact_long_timeseries_summary.json`
- windowed summary csv: `exact_long_windowed_summary.csv`
- raw timeseries dir: `exact_long_timeseries/`

## Spectral / Floquet proxy outputs

- summary json: `spectral_summary.json`
- top localized modes: `spectral_top_modes.csv`
- convergence and diagonal-ensemble comparison: `spectral_convergence.csv`

## What to look for next in the outputs

1. Does the PP-primary mask around the long-horizon point at phi/pi≈5/9 open into a nonzero interval under local refinement?
2. Does the high-advantage peak stay near the T=1500 long-horizon location rather than the original T=300 location?
3. In the spectral proxy, do the high-advantage points have larger overlap with top localized Floquet modes than the PP-primary point?
4. Do the diagonal-ensemble localization predictions stabilize as the ring size L increases and agree with direct ring time averages?
