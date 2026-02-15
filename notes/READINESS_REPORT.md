# Paper readiness report

Generated automatically by scripts/paper_readiness_report.py

## Latest run folders detected
- Stage0: outputs\run_20260210_124831_stage0_jan
- Stage1: outputs\run_20260210_135419_stage1_defect_phi
- Stage2: outputs\run_20260210_144641_stage2_pd_atlas

## Reproducibility file check (bootstrap_plan expects these per run)
- Stage0: OK (all expected files present)
- Stage1: OK (all expected files present)
- Stage2: OK (all expected files present)

## Stage0 key results (Jan-style replication, dev run)
- A: v_fit=-0.0462174, deltaP_late_mean=-0.0628356, w_loc=0.0249369, P0bar=0.00492315
- B: v_fit=-0.180007, deltaP_late_mean=-0.597338, w_loc=0.107233, P0bar=0.0210313
- ABB: v_fit=-0.0136878, deltaP_late_mean=0.102909, w_loc=0.357645, P0bar=0.0814691

## Stage1 key results (defect-only phi scan at p=0, dev run)
- ABB: phi_at_max_abs_v_fit=4.01426, max_abs_v_fit=-0.181879
- phi_at_phi_start=0, v_fit_at_phi_start=-0.012772, deltaP_late_mean_at_phi_start=0.084426

## Stage2 key results (PD atlas, dev run)
- Grid: Nphi=24, Np=9 (total=216)
- w_loc_ABB: min=0.141731, max=0.761569
- Directed transport thresholds: w_thr_primary=0.1, w_thr_sensitivity=0.15
- directed_transport_points_primary = 0
- directed_transport_points_sensitivity = 1
- strict_mask_points = 71
- parrondo_positive_points_primary = 0
- parrondo_positive_points_sensitivity = 1
- max_adv_v_max_ABB = 0.161175 at phi=1.0472 (phi_over_pi=0.333333), p=0
- max_v_fit_ABB = 0.137258 at phi=1.309 (phi_over_pi=0.416667), p=0
- min_w_loc_ABB = 0.141731 at phi=6.02139 (phi_over_pi=1.91667), p=0
- mismatch_sign(v_fit_ABB) vs sign(deltaP_late_mean_ABB): 91 / 216

## Confirmatory settings (from mrc.yaml, should match plan.txt Box MR)
- step_order: coin -> defect -> coin-noise -> shift
- T_long: 300, T0: 200, T1: 250
- stage3_grid: N_phi=72, N_p=31
- N_traj: 400, bootstrap_B: 1000, bootstrap_alpha: 0.05, base_seed: 20260125
- w_thr_primary: 0.1, w_thr_sensitivity: 0.15, x0_primary: 3, x0_sensitivity: 5

## What is still missing to finish the paper (bootstrap_plan minimal done)
- Fill in paper/paper.md (methods, results for 3 panels, limitations, references).
- Decide whether to run Milestone 8 robustness modules (bootstrap CIs, null baselines, region-level cluster overlay).
- If making region claims, run confirmatory Stage3 with the locked settings above (and save config_used.yaml + manifest.json).

