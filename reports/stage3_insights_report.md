# Stage-3 Insights Report

- run_dir: outputs\run_20260212_090815_stage3_pd_confirmatory
- grid: N_phi=72, N_p=31, total=2232
- sim: {
  "T": 300,
  "T0": 200,
  "T1": 250,
  "N_traj": 400,
  "pad_m": 2,
  "step_order": "coin -> defect -> coin-noise -> shift",
  "base_seed": 20260125
}
- thresholds: {
  "w_thr_primary": 0.1,
  "w_thr_sensitivity": 0.15,
  "x0_primary": 3,
  "x0_sensitivity": 5,
  "v_min": 0.008983509027373326,
  "eps_v": 0.009317532244377256,
  "eps_P": 0.08830678596863568
}

## Count sanity-check

metrics_summary.json counts:
```
{
  "strict_mask_points": 45,
  "directed_transport_points_primary": 6,
  "directed_transport_points_sensitivity": 8,
  "parrondo_positive_points_primary": 0,
  "parrondo_positive_points_sensitivity": 1,
  "mismatch_sign_points": 8,
  "mismatch_sign_denominator": 148
}
```
recomputed from atlas masks:
```
{
  "strict_mask_points": 45,
  "directed_transport_points_primary": 6,
  "directed_transport_points_sensitivity": 8,
  "parrondo_positive_points_primary": 0,
  "parrondo_positive_points_sensitivity": 1,
  "mismatch_sign_points": 8
}
```

## Headline maximum advantage point

```
{
  "phi_index": 12,
  "p_index": 0,
  "phi": 1.0471975511965976,
  "phi_over_pi": 0.3333333333333333,
  "p": 0.0,
  "v_fit_A": -0.05354750247179229,
  "v_fit_B": -0.03879155192411138,
  "v_fit_ABB": 0.12039081438979324,
  "adv_v": 0.15918236631390462,
  "w_loc_primary": 0.5539169355050061,
  "w_loc_sensitivity": 0.560667705564583,
  "P0bar": 0.17583604833778338,
  "deltaP_late_mean": 0.40298969393240414,
  "strict": 1,
  "dt_primary": 0,
  "dt_sens": 0,
  "pp_primary": 0,
  "pp_sens": 0,
  "mismatch": 0,
  "vA_ci_lo": -0.05354750247179229,
  "vA_ci_hi": -0.05354750247179229,
  "vB_ci_lo": -0.03879155192411138,
  "vB_ci_hi": -0.03879155192411138,
  "vABB_ci_lo": 0.12039081438979324,
  "vABB_ci_hi": 0.12039081438979324
}
```

## Parrondo-positive points

- primary count: 0
- sensitivity count: 1
- CSV: stage3_parrondo_points.csv

## Noise dependence quick check

- any PP_sens at p>0? False
- counts-by-p CSV: stage3_counts_by_p.csv

## Top-20 advantage points (CSV)

- stage3_top_adv_points.csv
