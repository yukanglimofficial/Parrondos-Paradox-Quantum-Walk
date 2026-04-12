from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


LABEL_ORDER = [
    'paper_max_T300',
    'refined_max_T6000',
    'pp_primary_mid_T6000',
    'pp_primary_left_T6000',
    'pp_primary_right_T6000',
]

LABEL_NAME = {
    'paper_max_T300': 'paper max ($\\phi/\\pi=1/3$)',
    'refined_max_T6000': 'refined max ($\\phi/\\pi\\approx0.295556$)',
    'pp_primary_mid_T6000': 'primary midpoint ($\\phi/\\pi\\approx0.558186$)',
    'pp_primary_left_T6000': 'primary left seed',
    'pp_primary_right_T6000': 'primary right seed',
}


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_close(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def copy_stage3_figures(repo_root: Path, stage3_run_dir: Path | None) -> Dict[str, List[str]]:
    out = {'copied': [], 'missing': []}
    target_dir = ensure_dir(repo_root / 'figures' / 'stage3')
    needed = [
        'S3_L1_v_fit_ABB.png', 'S3_L2_adv_v_max_ABB.png', 'S3_L5_mask_strict.png', 'S3_L9_w_loc_primary_ABB.png',
        'S3_L6_mask_directed_primary.png', 'S3_L7_parrondo_overlay_primary.png', 'S3_L11_mask_directed_sensitivity.png', 'S3_L13_parrondo_overlay_sensitivity.png',
        'S3_L12_deltaP_late_mean_ABB.png', 'S3_L8_mismatch_sign.png', 'S3_L10_P0bar_ABB.png',
        'S3_L3_sign_v_fit_A.png', 'S3_L4_sign_v_fit_B.png',
    ]
    if stage3_run_dir is None:
        return out
    source_dir = stage3_run_dir / 'figures'
    for name in needed:
        dst = target_dir / name
        if dst.exists():
            continue
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            out['copied'].append(name)
        else:
            out['missing'].append(name)
    return out


def plot_selected_timeseries(stage8_run_dir: Path, fig_dir: Path) -> None:
    ts_dir = stage8_run_dir / 'selected_point_timeseries'

    # Figure 1: x(t)/t for ABB at representative points.
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in LABEL_ORDER:
        path = ts_dir / f'timeseries_{label}_ABB.csv'
        if not path.exists():
            continue
        df = load_csv(path)
        df = df[df['t'] >= 1].copy()
        ax.plot(df['t'], df['x_mean'] / df['t'], label=LABEL_NAME.get(label, label))
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\langle X_t\rangle / t$')
    ax.set_title(r'Representative exact ABB drift profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_close(fig, fig_dir / 'S8_L1_xt_over_t_selected_points.pdf')

    # Figure 2: w_loc3(t) for ABB at representative points.
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in LABEL_ORDER:
        path = ts_dir / f'timeseries_{label}_ABB.csv'
        if not path.exists():
            continue
        df = load_csv(path)
        ax.plot(df['t'], df['w_loc3_inst'], label=LABEL_NAME.get(label, label))
    ax.set_xlabel('t')
    ax.set_ylabel(r'$w_{\mathrm{loc}}(t;3)$')
    ax.set_title(r'Representative exact ABB localization profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_close(fig, fig_dir / 'S8_L2_wloc3_selected_points.pdf')


def _interval_plot(df: pd.DataFrame, mask_name: str, out_path: Path, title: str) -> None:
    sub = df[df['mask_name'] == mask_name].copy()
    if sub.empty:
        raise ValueError(f'No rows for mask_name={mask_name}')
    rows = []
    for policy in ['fixed', 'scaled']:
        left = sub[(sub['policy'] == policy) & (sub['side'] == 'left')]
        right = sub[(sub['policy'] == policy) & (sub['side'] == 'right')]
        if left.empty or right.empty:
            continue
        left_v = float(left.iloc[0]['phi_root_over_pi'])
        right_v = float(right.iloc[0]['phi_root_over_pi'])
        mid = 0.5 * (left_v + right_v)
        half = 0.5 * (right_v - left_v)
        rows.append((policy, left_v, right_v, mid, half, int(left.iloc[0]['T'])))

    fig, ax = plt.subplots(figsize=(7, 2.8))
    yvals = np.arange(len(rows))
    for y, (policy, left_v, right_v, mid, half, T) in zip(yvals, rows):
        ax.errorbar(mid, y, xerr=half, fmt='o', capsize=4)
        ax.text(right_v + 0.0015, y, rf'[{left_v:.6f}, {right_v:.6f}]$\pi$', va='center', fontsize=8)
    ax.set_yticks(yvals)
    ax.set_yticklabels([f'{policy}, $T={T}$' for policy, _, _, _, _, T in rows])
    ax.set_xlabel(r'$\phi/\pi$')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    save_close(fig, out_path)


def plot_boundaries(stage8_run_dir: Path, fig_dir: Path) -> None:
    df = load_csv(stage8_run_dir / 'boundary_continuation.csv')
    _interval_plot(df, 'pp_primary', fig_dir / 'S8_L3_boundary_continuation_primary.pdf', r'Primary boundary continuation near $5\pi/9$')
    _interval_plot(df, 'pp_sensitivity', fig_dir / 'S8_L4_boundary_continuation_sensitivity.pdf', r'Sensitivity boundary continuation near $5\pi/9$')


def plot_spectral(stage8_run_dir: Path, fig_dir: Path) -> None:
    conv = load_csv(stage8_run_dir / 'spectral_subspace_convergence.csv')
    proj = load_csv(stage8_run_dir / 'spectral_projector_decomposition.csv')

    # Largest-L slice
    max_L = int(conv['L'].max())
    convL = conv[conv['L'] == max_L].copy()
    convL['label_order'] = convL['label'].map({lab: i for i, lab in enumerate(LABEL_ORDER)})
    convL = convL.sort_values(['label_order'])

    # Figure 5: top-k overlap at largest L.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([LABEL_NAME.get(x, x) for x in convL['label']], convL['topk_overlap_weight'])
    ax.set_ylabel('top-16 localized-mode overlap')
    ax.set_title(r'Localized-mode overlap at $L=%d$' % max_L)
    ax.tick_params(axis='x', labelrotation=20)
    ax.grid(True, axis='y', alpha=0.3)
    save_close(fig, fig_dir / 'S8_L5_spectral_overlap_L300.pdf')

    # Figure 6: diagonal vs direct-ring localization at largest L.
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(convL['diag_period_avg_w_loc3'], convL['ring_direct_period_avg_w_loc3'])
    lo = min(convL['diag_period_avg_w_loc3'].min(), convL['ring_direct_period_avg_w_loc3'].min())
    hi = max(convL['diag_period_avg_w_loc3'].max(), convL['ring_direct_period_avg_w_loc3'].max())
    pad = 0.01 * max(1.0, hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle='--')
    for _, row in convL.iterrows():
        ax.annotate(LABEL_NAME.get(row['label'], row['label']), (row['diag_period_avg_w_loc3'], row['ring_direct_period_avg_w_loc3']), fontsize=7)
    ax.set_xlabel(r'diagonal-ensemble $\overline{w}_{\mathrm{loc}}(|x|\leq 3)$')
    ax.set_ylabel(r'direct-ring $\overline{w}_{\mathrm{loc}}(|x|\leq 3)$')
    ax.set_title(r'Diagonal vs direct-ring localization at $L=%d$' % max_L)
    ax.grid(True, alpha=0.3)
    save_close(fig, fig_dir / 'S8_L6_diag_vs_ring_L300.pdf')

    # Figure 7: projector decomposition fractions at largest L.
    projL = proj[proj['L'] == max_L].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label in LABEL_ORDER:
        sub = projL[projL['label'] == label].sort_values('k_localized_modes')
        if sub.empty:
            continue
        frac = sub['absolute_contribution_w_loc3'] / sub['full_diag_w_loc3']
        ax.plot(sub['k_localized_modes'], frac, marker='o', label=LABEL_NAME.get(label, label))
    ax.set_xlabel('k localized modes')
    ax.set_ylabel(r'fraction of full diagonal-ensemble $\overline{w}_{\mathrm{loc}}(|x|\leq 3)$')
    ax.set_title(r'Localized-subspace projector decomposition at $L=%d$' % max_L)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_close(fig, fig_dir / 'S8_L7_projector_decomposition_L300.pdf')

    # Figure 8: weighted tail-kappa at largest L from k=max projector row.
    kmax = int(projL['k_localized_modes'].max())
    tailL = projL[projL['k_localized_modes'] == kmax].copy()
    tailL['label_order'] = tailL['label'].map({lab: i for i, lab in enumerate(LABEL_ORDER)})
    tailL = tailL.sort_values('label_order')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([LABEL_NAME.get(x, x) for x in tailL['label']], tailL['weighted_tail_kappa'])
    ax.set_ylabel(r'weighted tail-decay summary $\kappa$')
    ax.set_title(r'Weighted tail-decay summary at $L=%d$' % max_L)
    ax.tick_params(axis='x', labelrotation=20)
    ax.grid(True, axis='y', alpha=0.3)
    save_close(fig, fig_dir / 'S8_L8_tail_kappa_L300.pdf')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate missing paper figures from existing Stage-8 outputs.')
    p.add_argument('--repo_root', default='.', help='Repository root where figures/stage8 should be written.')
    p.add_argument('--stage8_run_dir', required=True, help='Path to Stage-8 run directory.')
    p.add_argument('--stage3_run_dir', default=None, help='Optional Stage-3 run directory to backfill figures/stage3.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    stage8_run_dir = Path(args.stage8_run_dir).resolve()
    stage3_run_dir = Path(args.stage3_run_dir).resolve() if args.stage3_run_dir else None

    fig_dir = ensure_dir(repo_root / 'figures' / 'stage8')
    stage3_copy = copy_stage3_figures(repo_root, stage3_run_dir)
    plot_selected_timeseries(stage8_run_dir, fig_dir)
    plot_boundaries(stage8_run_dir, fig_dir)
    plot_spectral(stage8_run_dir, fig_dir)

    manifest = {
        'repo_root': str(repo_root),
        'stage8_run_dir': str(stage8_run_dir),
        'stage3_run_dir': str(stage3_run_dir) if stage3_run_dir else None,
        'generated_figures': sorted(p.name for p in fig_dir.glob('S8_*.pdf')),
        'stage3_copy': stage3_copy,
    }
    out = stage8_run_dir / 'figure_generation_manifest.json'
    out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
