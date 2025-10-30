"""
Unified ablation runner (superset agent by default).
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from forward_algorithms.simulate_expert_data import Simulation
from inverse_opt_stopping.iq_learn_unified import IQ_Agent, IQ_AgentSuperset


@dataclass
class AblationConfig:
    label: str
    kwargs: Dict[str, Any]


def _discretizer_if_needed(conservative: bool):
    if not conservative:
        return None
    from binning import CART_Discretizer
    return CART_Discretizer()


def model_grid(conservative: bool, q_lr: float, env_lr: float, g_lr: float, eps: float, smote_k: int, is_cs: bool) -> List[AblationConfig]:
    base = dict(q_lr=q_lr, env_lr=env_lr, g_lr=g_lr, epsilon=eps, divergence='hellinger',
                cross_val_splits=(1 if conservative else 2),
                conservative=conservative, SMOTE_K=smote_k, is_cs=is_cs)
    return [
        AblationConfig('Classifier', dict(**base, classify=True, approx_g=False, approx_dynamics=False, oversampling=False)),
        AblationConfig('Classifier-SMOTE', dict(**base, classify=True, approx_g=False, approx_dynamics=False, oversampling='SMOTE')),
        AblationConfig('IQS', dict(**base, classify=False, approx_g=False, approx_dynamics=False, oversampling='SMOTE')),
        AblationConfig('IQS-SMOTE', dict(**base, classify=False, approx_g=False, approx_dynamics=False, oversampling='SMOTE')),
        AblationConfig('IQS-CS-SMOTE', dict(**base, classify=False, approx_g=False, approx_dynamics=False, oversampling='CS-SMOTE')),
        AblationConfig('Model-based IQS', dict(**base, classify=False, approx_g=False, approx_dynamics=True, oversampling=None)),
        AblationConfig('Model-based IQS-SMOTE', dict(**base, classify=False, approx_g=False, approx_dynamics=True, oversampling='SMOTE')),
        AblationConfig('Model-based IQS-CS-SMOTE', dict(**base, classify=False, approx_g=False, approx_dynamics=True, oversampling='CS-SMOTE')),
        AblationConfig('DO-IQS', dict(**base, classify=False, approx_g=True, approx_dynamics=True, oversampling=None)),
        AblationConfig('DO-IQS-LB', dict(**base, classify=False, approx_g=True, approx_dynamics=True, oversampling='CS-LSMOTE')),
    ]


def _ensure_best_snapshots(agent) -> None:
    """Ensure delegate/impl has best_q_net/best_g_net snapshots for testing.

    Some classifier training paths may not populate best_* fields.
    This function sets them to the current state_dicts if absent.
    """
    impl = getattr(agent, 'impl', agent)
    try:
        if not hasattr(impl, 'best_q_net') or impl.best_q_net is None:
            impl.best_q_net = impl.q_net.state_dict()
        if hasattr(impl, 'g_net') and impl.g_net is not None:
            if not hasattr(impl, 'best_g_net') or impl.best_g_net is None:
                impl.best_g_net = impl.g_net.state_dict()
    except Exception:
        pass


def simulate_example(example: str, episodes_train: int = 250, episodes_test: int = 100, max_path_length: int = 100) -> Tuple[dict, dict, int]:
    problem = {
        'radial': 'radial', 'star': 'star',
        'CP1': 'CP1', 'CP2': 'CP2', 'CP3': 'CP3', 'CP1M': 'CP1M', 'CP2M': 'CP2M', 'CP3M': 'CP3M',
        'bmG': 'symm_bm_G', 'bmgG': 'symm_bm_gG', 'car': 'car', 'azure': 'azure', 'nasa_turbofan': 'nasa_turbofan',
    }[example]
    t_downsample = 10 if example in {'azure', 'nasa_turbofan'} else 1
    sim = Simulation(problem=problem, total_n=episodes_train)
    train = sim.simulate_expert(episodes=episodes_train, max_path_length=max_path_length)
    test = sim.simulate_test(episodes=episodes_test, max_path_length=max_path_length)
    return train, test, t_downsample


def run_ablation_for_example(example: str, conservative: bool, seeds: List[int], n_epochs: int, batch_size: int,
                             out_thresh: float, out_thresh_alt: float, q_lr: float, env_lr: float, g_lr: float,
                             eps: float, smote_k: int, is_cs: bool, legacy: bool):
    os.makedirs('outputs', exist_ok=True)
    BM_BALANCED_ACC: List[List[float]] = []
    BM_MTTE: List[List[float]] = []
    BM_MEMR: List[List[float]] = []

    train, test, t_downsample = simulate_example(example)
    grid = model_grid(conservative, q_lr, env_lr, g_lr, eps, smote_k, is_cs)
    disc = _discretizer_if_needed(conservative)
    # Fit discretizer upfront for conservative flows so that `.ints` is available
    if conservative and disc is not None and hasattr(disc, 'fit'):
        try:
            disc.fit(np.asarray(train['state_mem']), np.asarray(train['action_mem']))
        except Exception:
            pass

    for s in seeds:
        np.random.seed(s)
        row_ba: List[float] = []
        row_mtte: List[float] = []
        row_memr: List[float] = []
        for cfg in grid:
            if legacy:
                variant = 'conserv' if conservative else 'base'
                agent = IQ_Agent(
                    obs_dim=train['state_mem'][0].shape[0],
                    action_dim=2,
                    variant=variant,
                    discretiser=disc,
                    **cfg.kwargs,
                )
            else:
                # Use superset with optional pre-oversampling. Native trainer is used
                # for classifier paths; other paths delegate to canonical for now.
                agent = IQ_AgentSuperset(
                    obs_dim=train['state_mem'][0].shape[0],
                    action_dim=2,
                    discretiser=disc,
                    pre_oversample=True,
                    use_native=True,
                    **cfg.kwargs,
                )
            agent.train(mem=train, batch_size=batch_size, n_epoches=n_epochs)
            _ensure_best_snapshots(agent)
            agent.out_thresh = out_thresh
            agent.test(test_memory=test, from_grid=False)
            row_ba.append(agent.balanced_accuracy)
            row_mtte.append(agent.mtte)
            row_memr.append(agent.memr)
            agent.out_thresh = out_thresh_alt
            agent.test(test_memory=test, from_grid=False)
            row_ba.append(agent.balanced_accuracy)
            row_mtte.append(agent.mtte)
            row_memr.append(agent.memr)
        BM_BALANCED_ACC.append(row_ba)
        BM_MTTE.append(row_mtte)
        BM_MEMR.append(row_memr)

    suf = '_fix_conserv' if conservative else ''
    np.save(f'outputs/{example}_balanced_acc{suf}.npy', np.array(BM_BALANCED_ACC))
    np.save(f'outputs/{example}_mtte{suf}.npy', np.array(BM_MTTE) * t_downsample)
    np.save(f'outputs/{example}_memr{suf}.npy', np.array(BM_MEMR))


def _summarize_mean_std(arr: np.ndarray) -> List[str]:
    m = arr.mean(0)
    s = arr.std(0)
    # mean +/- std for compact table display
    return [f"{m[i]:.4f} +/- {s[i]:.2f}" for i in range(arr.shape[1])]


def _summarize_median_iqr(arr: np.ndarray):
    med = np.median(arr, 0)
    q25 = np.quantile(arr, 0.25, 0)
    q75 = np.quantile(arr, 0.75, 0)
    return med, med - q25, q75 - med


def save_tradeoff_plot(example: str, suf: str, mtte: np.ndarray, memr: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    y_med, y_low, y_high = _summarize_median_iqr(mtte)
    x_med, x_low, x_high = _summarize_median_iqr(memr)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.get_cmap('tab10').colors
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '>', '<', '*']
    base_labels = ['Classifier','Classifier-SMOTE','IQS','IQS-SMOTE','IQS-CS-SMOTE','Model-based IQS','Model-based IQS-SMOTE','Model-based IQS-CS-SMOTE','DO-IQS','DO-IQS-LB']
    n = len(x_med)
    if n == 2 * len(base_labels):
        labels = [base_labels[i//2] for i in range(n)]
    else:
        labels = base_labels[:n]
    for i in range(n):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        ax.errorbar(x_med[i], y_med[i], yerr=[[y_low[i]], [y_high[i]]], xerr=[[x_low[i]], [x_high[i]]],
                    fmt=m, color=c, ecolor=(0.4, 0.4, 0.4, 0.6), elinewidth=1, capsize=3, mec='black', mfc=c, ms=6,
                    label=labels[i])
    ax.set_xlabel('MEMR')
    ax.set_ylabel('MTTE')
    ax.set_title(f'{example} - MTTE vs MEMR ({"Conservative" if suf else "Base"})')
    ax.legend(loc='best', fontsize=8, frameon=True)
    plt.savefig(os.path.join(out_dir, f'{example}{suf}_mtte_memr.png'), dpi=200)
    plt.close(fig)


def save_tables_txt(example: str, suf: str, ba: np.ndarray, mtte: np.ndarray, memr: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    base_labels = ['Classifier','Classifier-SMOTE','IQS','IQS-SMOTE','IQS-CS-SMOTE','Model-based IQS','Model-based IQS-SMOTE','Model-based IQS-CS-SMOTE','DO-IQS','DO-IQS-LB']
    n = ba.shape[1]
    if n == 2 * len(base_labels):
        headers = [base_labels[i//2] for i in range(n)]
    else:
        headers = base_labels[:n]
    lines = [f'Tables for {example} {"Conservative" if suf else "Base"}', '', '\t' + '\t'.join(headers)]
    for name, arr in [('BA', ba), ('MTTE', mtte), ('MEMR', memr)]:
        lines.append(name + '\t' + '\t'.join(_summarize_mean_std(arr)))
    with open(os.path.join(out_dir, f'{example}{suf}_tables.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def save_tables_tex(example: str, suf: str, ba: np.ndarray, mtte: np.ndarray, memr: np.ndarray, out_dir: str):
    """Save LaTeX table with model names and mean +/- std for BA/MTTE/MEMR."""
    os.makedirs(out_dir, exist_ok=True)
    # Labels ordered as grid configuration
    labels = ['Classifier','Classifier-SMOTE','IQS','IQS-SMOTE','IQS-CS-SMOTE',
              'Model-based IQS','Model-based IQS-SMOTE','Model-based IQS-CS-SMOTE','DO-IQS','DO-IQS-LB'][:ba.shape[1]]
    m_ba, s_ba = ba.mean(0), ba.std(0)
    m_mtte, s_mtte = mtte.mean(0), mtte.std(0)
    m_memr, s_memr = memr.mean(0), memr.std(0)
    lines = []
    lines.append('\\begin{table}[h]')
    lines.append('\\centering')
    lines.append(f"\\caption{{{example} {'Conservative' if suf else 'Base'}: BA/MTTE/MEMR (mean $\\pm$ std)}}")
    lines.append('\\begin{tabular}{lccc}')
    lines.append('\\toprule')
    lines.append('Model & BA & MTTE & MEMR \\\\')
    lines.append('\\midrule')
    for i, name in enumerate(labels):
        lines.append("{} & {:.4f} \\pm {:.2f} & {:.4f} \\pm {:.2f} & {:.4f} \\pm {:.2f} \\\\".format(name, m_ba[i], s_ba[i], m_mtte[i], s_mtte[i], m_memr[i], s_memr[i]))
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    with open(os.path.join(out_dir, f"{example}{suf}_tables.tex"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def parse_args():
    p = argparse.ArgumentParser(description='Unified ablation runner')
    p.add_argument('--examples', nargs='+', default=['azure'], help='Examples to run')
    p.add_argument('--seeds', type=int, default=1, help='Number of seeds')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--q-lr', type=float, default=0.01)
    p.add_argument('--env-lr', type=float, default=0.01)
    p.add_argument('--g-lr', type=float, default=0.001)
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--smote-k', type=int, default=12)
    p.add_argument('--is-cs', action='store_true')
    p.add_argument('--out-thresh', type=float, default=0.0)
    p.add_argument('--out-thresh-alt', type=float, default=0.005)
    p.add_argument('--base', action='store_true', help='Run non-conservative set (saves without suffix)')
    p.add_argument('--conservative', action='store_true', help='Run conservative set (saves with _fix_conserv)')
    p.add_argument('--legacy', action='store_true', help='Use legacy wrapper instead of superset agent')
    p.add_argument('--plots-dir', type=str, default='outputs/ablation', help='Directory to save plots and tables')
    p.add_argument('--no-plots', action='store_true', help='Disable saving MTTE/MEMR plots')
    p.add_argument('--no-tables', action='store_true', help='Disable saving tables to text files')
    p.add_argument('--print-tables', action='store_true', help='Print meanÂ+/-std tables to stdout')
    return p.parse_args()


def main():
    args = parse_args()
    seeds = list(range(args.seeds))
    run_base = args.base or (not args.base and not args.conservative)
    run_conserv = args.conservative or (not args.base and not args.conservative)

    for ex in args.examples:
        if run_base:
            run_ablation_for_example(ex, False, seeds, args.epochs, args.batch_size, args.out_thresh, args.out_thresh_alt,
                                     args.q_lr, args.env_lr, args.g_lr, args.eps, args.smote_k, args.is_cs, args.legacy)
        if run_conserv:
            run_ablation_for_example(ex, True, seeds, args.epochs, args.batch_size, args.out_thresh, args.out_thresh_alt,
                                     args.q_lr, args.env_lr, args.g_lr, args.eps, args.smote_k, args.is_cs, args.legacy)

        # Post-process plots/tables
        for suf, title in [('', 'BASE'), ('_fix_conserv', 'CONSERVATIVE')]:
            ba_p = f'outputs/{ex}_balanced_acc{suf}.npy'
            mtte_p = f'outputs/{ex}_mtte{suf}.npy'
            memr_p = f'outputs/{ex}_memr{suf}.npy'
            if not (os.path.exists(ba_p) and os.path.exists(mtte_p) and os.path.exists(memr_p)):
                continue
            ba = np.load(ba_p)
            mtte = np.load(mtte_p)
            memr = np.load(memr_p)
            if not args.no_plots:
                save_tradeoff_plot(ex, suf, mtte, memr, args.plots_dir)
            if args.print_tables:
                from tabulate import tabulate
                base_labels = ['Classifier','Classifier-SMOTE','IQS','IQS-SMOTE','IQS-CS-SMOTE','Model-based IQS','Model-based IQS-SMOTE','Model-based IQS-CS-SMOTE','DO-IQS','DO-IQS-LB']
                ncols = ba.shape[1]
                if ncols == 2 * len(base_labels):
                    headers = [base_labels[i//2] for i in range(ncols)]
                else:
                    headers = base_labels[:ncols]
                rows = [_summarize_mean_std(ba), _summarize_mean_std(mtte), _summarize_mean_std(memr)]
                print(f'== {ex} {title} ==')
                print(tabulate([[name] + row for name, row in zip(['BA','MTTE','MEMR'], rows)], headers=['Metric']+headers, tablefmt='github'))
            if not args.no_tables:
                save_tables_txt(ex, suf, ba, mtte, memr, args.plots_dir)
                save_tables_tex(ex, suf, ba, mtte, memr, args.plots_dir)


if __name__ == '__main__':
    main()





