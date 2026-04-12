# Stage-7 paper patch notes

## What changed relative to the current PDF

1. The locked Stage-3 atlas statement remains true for T=300, but the exact long-horizon noiseless follow-up no longer supports the blanket phrase 'no primary PP point'.
2. Stage 6 already opened a nonzero primary PP interval near phi/pi≈5/9; Stage 7 should be used to quote interval endpoints rather than a single coarse-grid point.
3. The strongest drift-advantage location is not exactly phi/pi=1/3 in the exact long-horizon/refined analysis; it sits in a nearby localized band around phi/pi≈0.296.

## Suggested revised thesis sentence

> For the Jan coins, an origin phase defect produces an exact drift-based Parrondo reversal in the noiseless DTQW. At long exact horizons, a narrow primary directed-transport interval emerges near phi/pi≈5/9, but the largest drift-advantage region remains strongly localized and is therefore distinct from robust transport.

## Numbers to update in the Results section

### Exact coarse-grid counts
```json
[
  {
    "policy": "fixed",
    "T": 3000,
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
    "max_adv_v": 0.15927822966672492,
    "max_adv_w_loc3": 0.5733153716524112,
    "max_adv_w_loc5": 0.5796209365897839,
    "max_adv_v_fit_ABB": 0.11154356621920736
  },
  {
    "policy": "fixed",
    "T": 6000,
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
    "max_adv_v": 0.15928025434174353,
    "max_adv_w_loc3": 0.5733147530896847,
    "max_adv_w_loc5": 0.5796190222921888,
    "max_adv_v_fit_ABB": 0.11154394448070569
  },
  {
    "policy": "scaled",
    "T": 3000,
    "T0": 2000,
    "T1": 2500,
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
    "max_adv_v": 0.1592658846034482,
    "max_adv_w_loc3": 0.5733148465339079,
    "max_adv_w_loc5": 0.5796169666508406,
    "max_adv_v_fit_ABB": 0.11154378869499611
  },
  {
    "policy": "scaled",
    "T": 6000,
    "T0": 4000,
    "T1": 5000,
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
    "max_adv_v": 0.1592808631739511,
    "max_adv_w_loc3": 0.5733164134062247,
    "max_adv_w_loc5": 0.5796172233376731,
    "max_adv_v_fit_ABB": 0.11154447504772702
  }
]
```

### Primary and sensitivity interval estimates near phi/pi≈5/9
```json
{
  "pp_primary": {
    "fixed": {
      "3000": {
        "left_phi": 1.7328209169864455,
        "right_phi": 1.7742000487396605,
        "left_phi_over_pi": 0.5515740288628472,
        "right_phi_over_pi": 0.5647454155815972,
        "width": 0.04137913175321506,
        "width_over_pi": 0.013171386718749964,
        "midpoint_phi": 1.753510482863053,
        "midpoint_phi_over_pi": 0.5581597222222222,
        "strict_stage3_all_profile_points": 1,
        "drift_stage3_all_profile_points": 1,
        "pp_primary_all_profile_points": 1,
        "pp_sensitivity_all_profile_points": 1
      },
      "6000": {
        "left_phi": 1.733178845836952,
        "right_phi": 1.7740083011411747,
        "left_phi_over_pi": 0.5516879611545139,
        "right_phi_over_pi": 0.5646843804253472,
        "width": 0.04082945530422277,
        "width_over_pi": 0.01299641927083332,
        "midpoint_phi": 1.7535935734890633,
        "midpoint_phi_over_pi": 0.5581861707899305,
        "strict_stage3_all_profile_points": 1,
        "drift_stage3_all_profile_points": 1,
        "pp_primary_all_profile_points": 1,
        "pp_sensitivity_all_profile_points": 1
      }
    },
    "scaled": {
      "3000": {
        "left_phi": 1.7324502049627062,
        "right_phi": 1.773675938637133,
        "left_phi_over_pi": 0.5514560275607638,
        "right_phi_over_pi": 0.564578586154514,
        "width": 0.041225733674426834,
        "width_over_pi": 0.013122558593750073,
        "midpoint_phi": 1.7530630717999196,
        "midpoint_phi_over_pi": 0.5580173068576388,
        "strict_stage3_all_profile_points": 1,
        "drift_stage3_all_profile_points": 1,
        "pp_primary_all_profile_points": 1,
        "pp_sensitivity_all_profile_points": 1
      },
      "6000": {
        "left_phi": 1.7335623410339231,
        "right_phi": 1.7737270713300626,
        "left_phi_over_pi": 0.5518100314670138,
        "right_phi_over_pi": 0.5645948621961806,
        "width": 0.04016473029613943,
        "width_over_pi": 0.012784830729166792,
        "midpoint_phi": 1.7536447061819929,
        "midpoint_phi_over_pi": 0.5582024468315973,
        "strict_stage3_all_profile_points": 1,
        "drift_stage3_all_profile_points": 1,
        "pp_primary_all_profile_points": 1,
        "pp_sensitivity_all_profile_points": 1
      }
    }
  },
  "pp_sensitivity": {
    "fixed": {
      "3000": {
        "left_phi": 1.7245757702515603,
        "right_phi": 1.7988715730781546,
        "left_phi_over_pi": 0.5489495171440973,
        "right_phi_over_pi": 0.5725986056857639,
        "width": 0.07429580282659431,
        "width_over_pi": 0.023649088541666592,
        "midpoint_phi": 1.7617236716648574,
        "midpoint_phi_over_pi": 0.5607740614149306,
        "strict_stage3_all_profile_points": 1,
        "drift_stage3_all_profile_points": 1,
        "pp_primary_all_profile_points": 0,
        "pp_sensitivity_all_profile_points": 1
      },
      "6000": {
        "left_phi": 1.7251126635273202,
        "right_phi": 1.7987309581725983,
        "left_phi_over_pi": 0.5491204155815973,
        "right_phi_over_pi": 0.5725538465711806,
        "width": 0.07361829464527814,
        "width_over_pi": 0.02343343098958325,
        "midpoint_phi": 1.7619218108499592,
        "midpoint_phi_over_pi": 0.5608371310763889,
        "strict_stage3_all_profile_points": 1,
        "drift_stage3_all_profile_points": 1,
        "pp_primary_all_profile_points": 0,
        "pp_sensitivity_all_profile_points": 1
      }
    },
    "scaled": {
      "3000": {
        "left_phi": 1.7255600745904534,
        "right_phi": 1.7985903432670425,
        "left_phi_over_pi": 0.5492628309461806,
        "right_phi_over_pi": 0.5725090874565973,
        "width": 0.07303026867658913,
        "width_over_pi": 0.02324625651041674,
        "midpoint_phi": 1.762075208928748,
        "midpoint_phi_over_pi": 0.560885959201389,
        "strict_stage3_all_profile_points": 1,
        "drift_stage3_all_profile_points": 1,
        "pp_primary_all_profile_points": 0,
        "pp_sensitivity_all_profile_points": 1
      },
      "6000": {
        "left_phi": 1.725547291417221,
        "right_phi": 1.79857756009381,
        "left_phi_over_pi": 0.549258761935764,
        "right_phi_over_pi": 0.5725050184461806,
        "width": 0.07303026867658913,
        "width_over_pi": 0.02324625651041674,
        "midpoint_phi": 1.7620624257555155,
        "midpoint_phi_over_pi": 0.5608818901909722,
        "strict_stage3_all_profile_points": 1,
        "drift_stage3_all_profile_points": 1,
        "pp_primary_all_profile_points": 0,
        "pp_sensitivity_all_profile_points": 1
      }
    }
  }
}
```

### Refined maximum-advantage estimates
```json
[
  {
    "policy": "fixed",
    "T": 6000,
    "phi": 0.9285151620609833,
    "phi_over_pi": 0.29555555555555557,
    "adv_v": 0.16047158883915852,
    "v_fit_A": -0.05231722992156076,
    "v_fit_B": -0.052331126635995664,
    "v_fit_ABB": 0.10815435891759777,
    "w_loc3_ABB": 0.579483674013146,
    "w_loc5_ABB": 0.5857178221085625,
    "P0bar_ABB": 0.18717792147788073,
    "strict_stage3": 1,
    "drift_stage3": 1,
    "pp_primary": 0,
    "pp_sensitivity": 0
  },
  {
    "policy": "scaled",
    "T": 6000,
    "phi": 0.9288642279113821,
    "phi_over_pi": 0.29566666666666663,
    "adv_v": 0.16047276588744006,
    "v_fit_A": -0.052320820207112845,
    "v_fit_B": -0.052279895479411455,
    "v_fit_ABB": 0.1081928704080286,
    "w_loc3_ABB": 0.5794165367483124,
    "w_loc5_ABB": 0.5856502255845354,
    "P0bar_ABB": 0.18722994090628517,
    "strict_stage3": 1,
    "drift_stage3": 1,
    "pp_primary": 0,
    "pp_sensitivity": 0
  }
]
```

## Suggested new subsection headings

1. Exact noiseless p=0 follow-up: horizon stability and interval structure
2. Transport interval near phi/pi≈5/9: threshold-defined boundaries from w_loc
3. Localization mechanism: finite-volume Floquet proxy and localized-mode overlap

## What to say in Discussion

- The pre-specified Stage-3 atlas is a finite-horizon statement.
- The long exact p=0 follow-up separates two phenomena: a broad, highly localized high-advantage band and a much narrower transport-permitting interval near 5/9 pi.
- The spectral proxy should be described as supporting, not proving, a localized-mode explanation.
