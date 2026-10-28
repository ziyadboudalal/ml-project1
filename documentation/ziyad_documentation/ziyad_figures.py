"""Reproduce four figures for Ziyad's CS-433 Project 1 experiments.

From repository root:
    python documentation/ziyad_figures.py

Prerequisites: submissions/ziyad.py, documentation/ziyad_results.json,
    dataset/x_train.csv, dataset/y_train.csv.

Default: regenerate plots using previously saved ablation results.
To rerun all ablation model training from scratch, add --recompute-ablation.
This script does not require ziyad_experiments.py or ziyad_ablation.csv.

Uses NumPy and matplotlib for visualization; no outside datasets.
Reports F1 on a 70/15/15 split, with cutoff chosen on tuning data.
"""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submissions"))
import ziyad
OUT = ROOT / "documentation" / "figures_ziyad"
RECORD = ROOT / "documentation" / "ziyad_results.json"
DATA = ROOT / "dataset"
INK = "#16253A"
BLUE = "#1668A9"
TEAL = "#088782"
ORANGE = "#D57926"
GRAY = "#6D7A89"
PAPER = "#FFFFFF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": GRAY,
    "ytick.color": GRAY, "axes.edgecolor": "#CBD3DD",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER})

def save(name):
    plt.savefig(OUT / name, dpi=240, bbox_inches='tight', facecolor=PAPER)
    plt.close()
    print(f'  saved {name}', flush=True)


def three_way_stratified(y, seed):
    """Disjoint stratified 70/15/15 train/tune/evaluation partitions."""
    train_tune, test = ziyad.split_stratified(y, fraction=0.15, seed=seed)
    tr_rel, tune_rel = ziyad.split_stratified(
        y[train_tune], fraction=0.15 / 0.85, seed=seed + 1000
    )
    train = train_tune[tr_rel]
    tune = train_tune[tune_rel]
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(tune, test).size
    assert not np.intersect1d(train, tune).size
    return train, tune, test


def one_ablation_experiment(x, y, names, variant, seed):
    """Recalculate an ablation result with no dependency on a separate script."""
    tr, tune, test = three_way_stratified(y, seed)
    encoder = ziyad.FeatureEncoder(variant).fit(x[tr], names)
    A = encoder.transform(x[tr])
    B = encoder.transform(x[tune])
    C = encoder.transform(x[test])
    w, epoch = ziyad.train_adam(
        A, y[tr], xv=B, yv=y[tune], epochs=12, l2=0.0002,
        seed=seed, verbose=False
    )
    p_tune = ziyad.sigmoid(B @ w)
    cutoff, tune_score = ziyad.best_threshold(y[tune], p_tune, metric='f1')
    p_test = ziyad.sigmoid(C @ w)
    test_score = ziyad.metrics(y[test], p_test >= cutoff)
    print(f'  {variant} seed={seed}: test F1={test_score[3]:.6f}',flush=True)
    return {'variant': variant, 'seed': seed, 'l2': 0.0002,
            'n_features': A.shape[1], 'selected_epoch': epoch,
            'threshold_from_tune': float(cutoff),
            'tune_f1': float(tune_score[3]),
            'test_accuracy': float(test_score[0]),
            'test_precision': float(test_score[1]),
            'test_recall': float(test_score[2]),
            'test_f1': float(test_score[3])}


def saved_ablation_runs():
    if not RECORD.exists():
        raise FileNotFoundError(
            'Missing documentation/ziyad_results.json. '
            'Use --recompute-ablation to recreate the ablation from scratch.'
        )
    previous = json.loads(RECORD.read_text(encoding='utf-8'))
    rows = previous.get('ablation_runs')
    if not rows:
        raise ValueError(
            'Saved JSON does not contain ablation_runs. '
            'Run with --recompute-ablation to calculate them.'
        )
    return rows

def plot_missingness(x):
    # Missing here means NaN/non-finite; coded nonresponse is a separate issue.
    missing = np.mean(~np.isfinite(x), axis=0)
    ordered = np.sort(missing)[::-1]
    n_half = int(np.count_nonzero(missing > 0.5))
    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    ax.fill_between(np.arange(len(ordered)), ordered, color=BLUE, alpha=.12)
    ax.plot(ordered, lw=1.45, color=BLUE)
    ax.axhline(.5,ls='--',lw=.8,color=ORANGE)
    ax.set(xlabel='Original features (sorted)',ylabel='Fraction missing',
           xlim=(0,len(ordered)-1),ylim=(0,1.03))
    ax.text(.97,.95,f'{n_half}/{len(ordered)} > 50% missing',ha='right',va='top',
            transform=ax.transAxes,fontsize=7.5)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save('missing_report.png')
    return {'features_over_50pct_missing': n_half,
            'mean_missing_fraction': float(missing.mean()),
            'features_over_90pct_missing': int(np.sum(missing > .9))}


def plot_ablation(rows):
    seeds = [42, 123, 2026]
    variants = ['expanded_no_interactions', 'expanded']
    labels = ['Encoded (no interactions)', 'Encoded + 7 interactions']
    data = {v: {} for v in variants}
    for r in rows:
        if r['variant'] in data:
            data[r['variant']][int(r['seed'])] = float(r['test_f1'])
    for v in variants:
        if set(data[v]) != set(seeds):
            raise ValueError(f'Ablation needs seeds {seeds} for {v}; got {sorted(data[v])}')
    score = np.array([[data[v][s] for s in seeds] for v in variants])
    means = score.mean(axis=1)
    std = score.std(axis=1, ddof=1)
    fig, ax = plt.subplots(figsize=(6.7, 2.65))
    xlocations = np.arange(len(variants))
    ax.barh(xlocations, means, color=[TEAL, BLUE], height=.48, alpha=.88)
    ax.errorbar(means, xlocations, xerr=std, fmt='none', ecolor=INK, capsize=4, lw=1.2)
    ax.set_yticks(xlocations, labels)
    ax.invert_yaxis()
    ax.set(xlabel='Untouched holdout F1 (three stratified splits)', xlim=(0, .49))
    ax.set_xticks([0,.1,.2,.3,.4], ['0.0','0.1','0.2','0.3','0.4'])
    for i in range(2):
        ax.text(means[i] + std[i] + .006, i,
                f'{means[i]:.4f} +/- {std[i]:.4f}', va='center', fontsize=9)
    ax.text(0.98, .05, 'Seeds: 42, 123, 2026  |  70/15/15 train/tune/test',
            transform=ax.transAxes, ha='right', fontsize=8, color=GRAY)
    fig.tight_layout()
    save('ablation.png')
    return {'seeds':seeds, 'without_interactions_mean_f1':float(means[0]),
            'with_interactions_mean_f1':float(means[1]),
            'without_interactions_sample_sd':float(std[0]),
            'with_interactions_sample_sd':float(std[1]),
            'paired_test_f1_difference':float(means[1]-means[0])}


def threshold_metrics(y, p, ts):
    yt = y.astype(bool)
    P = int(np.count_nonzero(yt))
    pp = p[:,None] >= ts[None,:]
    pred_count = pp.sum(axis=0)
    tp = np.sum(pp & yt[:,None], axis=0)
    precision = np.divide(tp,pred_count, out=np.zeros(ts.size),where=pred_count>0)
    recall = tp / P
    f1 = np.divide(2*tp, pred_count+P, out=np.zeros(ts.size),where=(pred_count+P)>0)
    return precision, recall, f1


def plot_threshold(y_tune, p_tune, threshold, y_test, p_test):
    grid = np.arange(.05,.7001,.005)
    precision, recall, f1 = threshold_metrics(y_tune,p_tune,grid)
    test_f1 = ziyad.metrics(y_test, p_test >= threshold)[3]
    # Paper-sized variant: readable at IEEE single-column width.
    fig, ax = plt.subplots(figsize=(3.35, 2.10))
    ax.plot(grid, f1, color=BLUE, lw=1.6, label='F1')
    ax.plot(grid, precision, color=ORANGE, lw=1.3, label='Precision')
    ax.plot(grid, recall, color=TEAL, lw=1.3, label='Recall')
    ax.axvline(threshold,ls='--',lw=1,color=INK)
    ax.axvline(.5,ls=':',lw=.9,color=GRAY)
    ax.set(xlabel='Classification cutoff',ylabel='Score',xlim=(.05,.7),ylim=(0,1.02))
    ax.legend(frameon=False,ncol=3,fontsize=7,loc='upper right')
    ax.text(threshold + .015, .03, f't = {threshold:.2f}',fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save('threshold_report.png')
    return {'tune_optimal_threshold':float(threshold),
            'test_f1_at_locked_threshold':float(test_f1),
            'test_f1_at_0_5':float(ziyad.metrics(y_test,p_test>=.5)[3]),
            'test_accuracy_at_locked_threshold':float(ziyad.metrics(y_test,p_test>=threshold)[0]),
            'test_precision_at_locked_threshold':float(ziyad.metrics(y_test,p_test>=threshold)[1]),
            'test_recall_at_locked_threshold':float(ziyad.metrics(y_test,p_test>=threshold)[2])}


def plot_top_weights(w, encoder, n=14):
    # Exclude bias. Coefficients are associative & depend on coding and L2.
    ids = np.argsort(np.abs(w[1:]))[-n:]+1
    ids = ids[np.argsort(np.abs(w[ids]))]
    names = np.asarray(encoder.feature_names,dtype=object)
    labels = [str(names[j-1]) for j in ids]
    v = w[ids]
    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    ax.barh(labels, v, color=[ORANGE if t>0 else BLUE for t in v], height=.64)
    ax.axvline(0,color=GRAY,linewidth=.8)
    ax.set_xlabel('Logistic coefficient (association only)',fontsize=8)
    ax.grid(axis='x')
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save('weights_report.png')
    return {'largest_coefficients': [{'feature':label,'weight':float(value)}
                                      for label,value in zip(labels,v)]}


def train_diagnostics(x,y,names):
    """Fit on 70%; choose model epoch/threshold on 15%; evaluate on last 15%."""
    tr,tune,test=three_way_stratified(y,seed=42)
    enc=ziyad.FeatureEncoder('expanded_no_interactions').fit(x[tr],names)
    A=enc.transform(x[tr]); B=enc.transform(x[tune]); C=enc.transform(x[test])
    print(f'Diagnostic partitions train/tune/test = {len(tr)}/{len(tune)}/{len(test)}',flush=True)
    w,ep=ziyad.train_adam(A,y[tr],xv=B,yv=y[tune],epochs=12,l2=.0002,seed=42,verbose=False)
    p_tune=ziyad.sigmoid(B @ w)
    p_test=ziyad.sigmoid(C @ w)
    cut,tune_metrics=ziyad.best_threshold(y[tune],p_tune,metric='f1')
    print('Selected epoch:',ep,'locked cutoff:',cut,'tune F1:',tune_metrics[3],flush=True)
    out={'training':{'seed':42,'train':len(tr),'tune':len(tune),'test':len(test),
                     'features':A.shape[1],'epochs':ep,'l2':.0002,
                     'tune_f1':float(tune_metrics[3])},
         'threshold':plot_threshold(y[tune],p_tune,cut,y[test],p_test),
         'coefficients':plot_top_weights(w,enc)}
    print('Untouched local test F1:',out['threshold']['test_f1_at_locked_threshold'],flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--recompute-ablation', action='store_true',
        help='Rerun model training for all ablation seeds instead of using saved results.'
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print('Loading training dataset...', flush=True)
    x, _, names = ziyad.read_features(DATA / 'x_train.csv')
    y = ziyad.load_labels(DATA / 'y_train.csv')
    if args.recompute_ablation:
        print('Recomputing ablation across seeds 42, 123, 2026...', flush=True)
        runs = [one_ablation_experiment(x, y, names, variant, seed)
                for seed in (42, 123, 2026)
                for variant in ('expanded_no_interactions', 'expanded')]
    else:
        print('Using saved ablation results (pass --recompute-ablation to retrain).',flush=True)
        runs = saved_ablation_runs()
    results = {
        'dataset': {'training_rows': len(y), 'features': len(names)},
        'ablation_runs': runs,
        'missingness': plot_missingness(x),
        'ablation': plot_ablation(runs),
        'diagnostic': train_diagnostics(x, y, names),
    }
    RECORD.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print('Figures completed and diagnostics saved to', RECORD, flush=True)


if __name__ == '__main__':
    main()
