# Artifact map

| Stage | Primary script(s) | Archived output directory | Notes |
|---|---|---|---|
| 0 | `scripts/stage0_jan_replication.py` | `outputs/run_20260210_124831_stage0_jan/` | Baseline replication used in the paper |
| 1 | `scripts/stage1_defect_phi_scan.py` | `outputs/run_20260210_135419_stage1_defect_phi/` | Exploratory deterministic defect-phase scan |
| 2 | `scripts/stage2_pd_atlas.py`, `scripts/stage2_postprocess_figures.py` | `outputs/run_20260210_144641_stage2_pd_atlas/` | Exploratory coarse noisy atlas |
| 3 | `scripts/stage3_confirmatory_pd_atlas_cpu.py`, `scripts/stage3_postprocess_figures.py`, `scripts/stage3_extract_insights.py` | `outputs/run_20260212_090815_stage3_pd_confirmatory/` | Confirmatory atlas used in the paper |
| 4 | `scripts/stage4_exact_p0_audit.py` | `outputs/run_20260322_094517_stage4_exact_p0_audit/` | Exact p=0 recomputation of the confirmatory slice |
| 5 | `scripts/stage5_rigour_suite_longrun.py` | `outputs/run_20260322_105844_stage5_rigour_suite_longrun/` | Deterministic continuation and numerical checks |
| 6 | `scripts/stage6_exact_p0_math_suite.py` | `outputs/run_20260322_125852_stage6_exact_p0_math_suite/` | Local refinement and early long-horizon exact study |
| 7 | `scripts/stage7_exact_p0_asymptotic_suite.py` | `outputs/run_20260322_161029_stage7_exact_p0_asymptotic_suite/` | Boundary continuation and asymptotic extrapolation |
| 8 | `scripts/stage8_rigour_bridge_suite.py`, `scripts/stage8_postprocess_figures.py` | `outputs/run_20260322_194423_stage8_rigour_bridge_suite/` | Longest-horizon selected points and spectral proxy analysis |

## Paper-facing figure roots

- `figures/stage3/`
- `figures/stage8/`

## Paper-facing report root

- `reports/`
