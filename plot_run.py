"""
Generate thesis figures from a single training run's logs.

Reads the CSVs that train_partseg.py writes into a run directory
(log/part_seg/<log_dir>/) and saves publication-style PNGs back into it:

  learning_curves.png     test acc / class mIoU / instance mIoU / train acc vs epoch
  train_loss_lr.png       train loss with the LR schedule overlaid
  per_category_miou.png    per-category mIoU bars (sorted) + instance/class avg lines
  miou_vs_samples.png      per-category mIoU vs trainval sample count (scatter)
  generalization_gap.png   train acc - test acc vs epoch
  summary.csv              best epoch + headline metrics + timing
  per_category_table.csv   category x {parts, trainval, test, mIoU}, sorted by mIoU

Pure stdlib + matplotlib (Agg backend), so it runs headless on the lab box.

Usage:
  python plot_run.py --log_dir 2026-06-09_14-59
  python plot_run.py --log_dir airplane_vs_car/Airplane/seed_42
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')  # headless: no display needed
import matplotlib.pyplot as plt

# Number of parts per category (from the ShapeNet seg_classes label ranges).
NUM_PARTS = {
    'Earphone': 3, 'Motorbike': 6, 'Rocket': 3, 'Car': 4, 'Laptop': 2,
    'Cap': 2, 'Skateboard': 3, 'Mug': 2, 'Guitar': 3, 'Bag': 2,
    'Lamp': 4, 'Table': 3, 'Airplane': 4, 'Chair': 4, 'Knife': 2,
}


def read_metrics(path):
    """metrics_per_epoch.csv -> dict of column_name -> list of floats."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        cols = {h: [] for h in header}
        for row in reader:
            if not row or not row[0].strip():
                continue
            for h, cell in zip(header, row):
                cols[h].append(float(cell.strip()))
    return cols


def read_per_category(path):
    """per_category_miou.csv -> list of (category, miou, n_test), sorted by miou desc."""
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            cat, miou, n = row[0].strip(), float(row[1]), int(row[2])
            rows.append((cat, miou, n))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def read_sample_counts(path):
    """sample_counts.csv -> dict category -> (trainval, test)."""
    counts = {}
    if not os.path.exists(path):
        return counts
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            counts[row[0].strip()] = (int(row[1]), int(row[2]))
    return counts


def lr_decay_epochs(epochs, lrs):
    """Epochs at which the learning rate dropped (for vertical markers)."""
    drops = []
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i - 1]:
            drops.append(epochs[i])
    return drops


def best_epoch(m):
    """Index/epoch of max instance-avg mIoU (matches best_model.pth selection)."""
    iou = m['test_instance_avg_iou']
    i = max(range(len(iou)), key=lambda k: iou[k])
    return i, int(m['epoch'][i])


def plot_learning_curves(m, out, decays):
    ep = m['epoch']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ep, m['test_instance_avg_iou'], label='Test instance-avg mIoU', lw=2)
    ax.plot(ep, m['test_class_avg_iou'], label='Test class-avg mIoU', lw=2)
    ax.plot(ep, m['test_accuracy'], label='Test accuracy', lw=2)
    ax.plot(ep, m['train_accuracy'], label='Train accuracy', lw=1.2, ls='--', alpha=0.8)
    for d in decays:
        ax.axvline(d, color='grey', ls=':', lw=0.8, alpha=0.6)
    if decays:
        ax.axvline(decays[0], color='grey', ls=':', lw=0.8, alpha=0.6, label='LR decay')
    bi, be = best_epoch(m)
    ax.scatter([be], [m['test_instance_avg_iou'][bi]], color='red', zorder=5,
               label='Best (epoch %d)' % be)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Metric')
    ax.set_title('Training and test metrics over epochs')
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_loss_lr(m, out):
    ep = m['epoch']
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(ep, m['train_loss'], color='tab:red', lw=2, label='Train loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red'); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ep, m['learning_rate'], color='tab:blue', lw=1.5, ls='--', label='Learning rate')
    ax2.set_ylabel('Learning rate', color='tab:blue')
    ax2.set_yscale('log'); ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Train loss and learning-rate schedule')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_per_category(rows, m, out):
    bi, _ = best_epoch(m)
    inst = m['test_instance_avg_iou'][bi]
    clas = m['test_class_avg_iou'][bi]
    cats = [r[0] for r in rows]
    ious = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, max(5, 0.45 * len(cats))))
    ypos = range(len(cats))
    bars = ax.barh(list(ypos), ious, color='tab:blue', alpha=0.85)
    ax.set_yticks(list(ypos)); ax.set_yticklabels(cats)
    ax.invert_yaxis()  # best at top
    ax.axvline(inst, color='tab:green', ls='--', lw=1.5, label='Instance-avg %.3f' % inst)
    ax.axvline(clas, color='tab:orange', ls='--', lw=1.5, label='Class-avg %.3f' % clas)
    for b, v in zip(bars, ious):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, '%.3f' % v,
                va='center', fontsize=8)
    ax.set_xlim(0, 1.0); ax.set_xlabel('Test mIoU (best epoch)')
    ax.set_title('Per-category mIoU')
    ax.legend(loc='lower right', fontsize=9); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_miou_vs_samples(rows, counts, out):
    pts = [(counts[c][0], iou, c) for c, iou, _ in rows if c in counts]
    if not pts:
        return False
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(xs, ys, color='tab:purple', s=40, zorder=3)
    for x, y, c in pts:
        ax.annotate(c, (x, y), textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Trainval samples (log scale)')
    ax.set_ylabel('Test mIoU')
    ax.set_title('Per-category mIoU vs training-set size')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return True


def plot_gap(m, out):
    ep = m['epoch']
    gap = [tr - te for tr, te in zip(m['train_accuracy'], m['test_accuracy'])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep, gap, color='tab:brown', lw=1.5)
    ax.axhline(0, color='grey', lw=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Train acc - Test acc')
    ax.set_title('Generalization gap over epochs')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def write_summary(m, rows, counts, exp_dir):
    bi, be = best_epoch(m)
    total_time = sum(m['epoch_time_seconds']) if 'epoch_time_seconds' in m else 0.0
    # epochs to within 0.5% of best instance mIoU
    best_iou = m['test_instance_avg_iou'][bi]
    converged = next((int(m['epoch'][k]) for k in range(len(m['epoch']))
                      if m['test_instance_avg_iou'][k] >= best_iou - 0.005), be)
    with open(os.path.join(exp_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['total_epochs', int(m['epoch'][-1])])
        w.writerow(['best_epoch', be])
        w.writerow(['best_instance_avg_iou', round(best_iou, 5)])
        w.writerow(['best_class_avg_iou', round(m['test_class_avg_iou'][bi], 5)])
        w.writerow(['accuracy_at_best', round(m['test_accuracy'][bi], 5)])
        w.writerow(['final_instance_avg_iou', round(m['test_instance_avg_iou'][-1], 5)])
        w.writerow(['max_test_accuracy', round(max(m['test_accuracy']), 5)])
        w.writerow(['epochs_to_within_0.5pct', converged])
        w.writerow(['total_train_time_hours', round(total_time / 3600.0, 2)])
        w.writerow(['mean_epoch_time_seconds', round(total_time / len(m['epoch']), 1)])

    with open(os.path.join(exp_dir, 'per_category_table.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'num_parts', 'trainval', 'test', 'miou'])
        for cat, iou, n_test in rows:
            tv = counts.get(cat, (None, n_test))[0]
            w.writerow([cat, NUM_PARTS.get(cat, ''), tv if tv is not None else '',
                        n_test, round(iou, 5)])
    return be, best_iou, converged, total_time


def main():
    ap = argparse.ArgumentParser('Plot a single training run')
    ap.add_argument('--log_dir', required=True,
                    help='run folder under log/part_seg/, e.g. 2026-06-09_14-59')
    ap.add_argument('--root', default='log/part_seg',
                    help='base dir containing the run folder')
    args = ap.parse_args()

    exp_dir = os.path.join(args.root, args.log_dir)
    metrics_path = os.path.join(exp_dir, 'metrics_per_epoch.csv')
    cat_path = os.path.join(exp_dir, 'per_category_miou.csv')
    counts_path = os.path.join(exp_dir, 'sample_counts.csv')

    if not os.path.exists(metrics_path):
        raise SystemExit('No metrics_per_epoch.csv in %s' % exp_dir)

    m = read_metrics(metrics_path)
    decays = lr_decay_epochs(m['epoch'], m['learning_rate'])
    counts = read_sample_counts(counts_path)

    plot_learning_curves(m, os.path.join(exp_dir, 'learning_curves.png'), decays)
    plot_loss_lr(m, os.path.join(exp_dir, 'train_loss_lr.png'))
    plot_gap(m, os.path.join(exp_dir, 'generalization_gap.png'))
    print('Saved: learning_curves.png, train_loss_lr.png, generalization_gap.png')

    if os.path.exists(cat_path):
        rows = read_per_category(cat_path)
        plot_per_category(rows, m, os.path.join(exp_dir, 'per_category_miou.png'))
        print('Saved: per_category_miou.png')
        if plot_miou_vs_samples(rows, counts, os.path.join(exp_dir, 'miou_vs_samples.png')):
            print('Saved: miou_vs_samples.png')
        be, best_iou, conv, tt = write_summary(m, rows, counts, exp_dir)
        print('Saved: summary.csv, per_category_table.csv')
        print('\nBest epoch %d  |  instance mIoU %.4f  |  converged ~epoch %d  |  %.1f h total'
              % (be, best_iou, conv, tt / 3600.0))
    else:
        print('No per_category_miou.csv -> skipped category figures/tables')


if __name__ == '__main__':
    main()
