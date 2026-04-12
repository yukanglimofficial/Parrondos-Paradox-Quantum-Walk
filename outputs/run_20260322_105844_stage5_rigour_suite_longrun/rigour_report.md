# Stage-5 Rigour Suite Report

## Stage-4 exact reference

```json
{
  "strict": 45,
  "drift_parrondo": 21,
  "pp_primary": 0,
  "pp_sensitivity": 1,
  "mismatch_sign": 1,
  "mismatch_sign_denominator": 50
}
```

Reference phis used in this suite:
```json
{
  "phi0": 0.0,
  "max_adv_phi": 1.0471975511965976,
  "pp_sens_phi": 1.7453292519943295
}
```

## Exact p=0 horizon sweep

- horizons tested: [300, 600, 900, 1200, 1500]
- counts csv: `exact_horizon_counts.csv`
- persistent stage3 csv: `persistent_points_stage3.csv`
- persistent raw csv: `persistent_points_raw.csv`

T=300 fixed-window count row:
```json
{
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
}
```

Largest requested horizon (fixed-window) count row:
```json
{
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
```

## Density-matrix validation

```json
{
  "n_points": 27,
  "T": 120,
  "T0": 80,
  "T1": 100,
  "validation_T": 120,
  "validation_T0": 80,
  "validation_T1": 100,
  "N_traj": 4000,
  "thresholds": {
    "abs_diff_v_fit": 0.01,
    "abs_diff_w_loc3": 0.02,
    "abs_diff_P0bar": 0.02
  },
  "n_pass_all": 27,
  "max_abs_diff_v_fit": 0.008046568741277344,
  "max_abs_diff_w_loc3": 0.005541809092443395,
  "max_abs_diff_P0bar": 0.0011753657368134135,
  "max_abs_diff_w_loc5": 0.0062508854837958205,
  "max_abs_diff_deltaP_late_mean": 0.020895031777673892,
  "csv": "density_validation.csv",
  "p1_phi_invariance_csv": "p1_phi_invariance.csv"
}
```

## Padding invariance validation

```json
{
  "n_points": 10,
  "N_traj": 400,
  "thresholds": {
    "abs_diff_v_fit": 0.005,
    "abs_diff_w_loc3": 0.01,
    "abs_diff_P0bar": 0.01
  },
  "n_pass_all": 10,
  "max_abs_diff_v_fit": 1.0265746586135549e-13,
  "max_abs_diff_w_loc3": 2.4702462297909733e-15,
  "max_abs_diff_P0bar": 1.8041124150158794e-16,
  "csv": "padding_validation.csv"
}
```

## Recommended writing use

1. Use the exact horizon sweep to decide whether your p=0 drift-Parrondo points persist or shrink with T.
2. Use the density-matrix validation table to justify the noisy trajectory engine as an implementation of the Kraus channel.
3. Use the padding validation table to close the protocol gap between mrc.yaml and the code actually executed.
4. Use p=1 phi-invariance from the density summary to support the claim that the defect phase is irrelevant after complete dephasing/classicalization.
