"""
Experiment driver: train SEPARATE single-category models (one category per run)
from scratch with FIXED hyperparameters, varying ONLY the category and the seed.

For each category in --categories, the full fixed seed list is run, paired by
position across categories (e.g. the "chair point net" and the "airplane point
net"). Total runs = len(categories) * len(seeds).

Calls train_partseg.py as a subprocess; does NOT reimplement training. The same
--subset_seed is used for every run; only the category and --seed change.
--cap is optional: pass an explicit integer for a balanced cap shared across
all categories (for fair cross-category comparison), or omit it entirely to
train each category on ALL of its available samples (no balancing).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Fixed pool of training seeds. Only this varies between runs.
SEEDS = [42, 7, 123, 2024, 99, 1, 256, 512, 1000, 31337]

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser('Experiment driver: repeated training over fixed seeds')
    parser.add_argument('--categories', nargs='+', required=True, help='categories to train separately, e.g. Airplane Car')
    parser.add_argument('--cap', default=None, help='common per-category sample count (explicit INTEGER, the pairwise minimum). '
                                                      'Omit to train each category on ALL of its available samples (no cross-category balancing).')
    parser.add_argument('--test_cap', type=int, default=None, help='cap TEST samples per category (default: no cap)')
    parser.add_argument('--full_parts_only', action='store_true', help='keep only objects with the full set of parts (train and test)')
    parser.add_argument('--num_parts', type=int, default=None, help='keep only objects with exactly this many distinct parts (train and test)')
    parser.add_argument('--subset_seed', type=int, default=0, help='FIXED subset RNG seed for the whole experiment')
    parser.add_argument('--epoch', type=int, required=True, help='epochs, fixed across all runs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size, fixed across all runs')
    parser.add_argument('--model', default='pointnet2_part_seg_msg', help='model name')
    parser.add_argument('--exp_name', type=str, required=True, help='top-level experiment folder under log/part_seg/')
    parser.add_argument('--max_seeds', type=int, default=None, help='use only the first N seeds (testing)')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device id to run all training subprocesses on, e.g. "0" or "1"')
    return parser.parse_args()


def main():
    args = parse_args()

    # Single-category runs need an explicit integer cap (no 'min'): each category
    # is trained alone, so 'min' would just equal that category's own size and
    # defeat the cross-category balancing. Omitting --cap entirely means "use
    # every available sample per category" (no balancing at all).
    if args.cap is None:
        cap = None
    else:
        try:
            cap = int(args.cap)
        except (TypeError, ValueError):
            both = ' '.join(args.categories)
            print(
                f"ERROR: --cap must be an explicit INTEGER for single-category runs, got {args.cap!r}.\n"
                f"       Each category trains alone, so 'min' can't be resolved across categories here.\n"
                f"       Find the common value by running once:\n"
                f"         python train_partseg.py --categories {both} --samples_per_category min\n"
                f"       and read off the printed cap (e.g. 740, set by Car), then pass --cap 740.",
                file=sys.stderr,
            )
            return 2

    seeds = SEEDS if args.max_seeds is None else SEEDS[:args.max_seeds]
    n_total = len(args.categories) * len(seeds)

    completed, skipped, failed = [], [], []

    print(f'Experiment "{args.exp_name}": {n_total} run(s) '
          f'({len(args.categories)} categories x {len(seeds)} seeds)')
    print(f'  categories={args.categories} (trained SEPARATELY)  '
          f'cap={cap if cap is not None else "None (all available samples per category)"}  '
          f'subset_seed={args.subset_seed} (FIXED)')
    print(f'  test_cap={args.test_cap}  full_parts_only={args.full_parts_only}  num_parts={args.num_parts}')
    print(f'  epoch={args.epoch}  batch_size={args.batch_size}  model={args.model}  gpu={args.gpu}')
    print(f'  seeds={seeds}')
    print('-' * 60)

    k = 0
    for category in args.categories:
        for seed in seeds:
            k += 1
            log_dir = f'{args.exp_name}/{category}/seed_{seed}'
            exp_dir = BASE_DIR / 'log' / 'part_seg' / args.exp_name / category / f'seed_{seed}'
            run_complete = exp_dir / 'RUN_COMPLETE'

            print(f'[{k}/{n_total}] category={category} seed={seed} -> log/part_seg/{log_dir}')

            if run_complete.exists():
                print(f'  SKIP already complete ({run_complete})')
                skipped.append((category, seed))
                continue

            # Not skipped: wipe any stale/partial run dir so training starts from scratch
            # and never resumes from a stale best_model.pth.
            if exp_dir.exists():
                print(f'  CLEAN stale run dir {exp_dir}')
                shutil.rmtree(exp_dir)

            # Pre-create the nested run dir (train_partseg.py uses mkdir without parents=True).
            exp_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, str(BASE_DIR / 'train_partseg.py'),
                '--categories', category,
                '--subset_seed', str(args.subset_seed),
                '--seed', str(seed),
                '--epoch', str(args.epoch),
                '--batch_size', str(args.batch_size),
                '--model', args.model,
                '--log_dir', log_dir,
                '--gpu', args.gpu,
            ]
            if cap is not None:
                cmd += ['--samples_per_category', str(cap)]
            if args.test_cap is not None:
                cmd += ['--test_samples_per_category', str(args.test_cap),
                        '--test_subset_seed', str(args.subset_seed)]
            if args.full_parts_only:
                cmd += ['--full_parts_only']
            if args.num_parts is not None:
                cmd += ['--num_parts', str(args.num_parts)]
            print(f'  RUN {" ".join(cmd[1:])}')

            result = subprocess.run(cmd, cwd=str(BASE_DIR))
            if result.returncode == 0:
                run_complete.touch()  # completion sentinel
                print(f'  DONE category={category} seed={seed}')
                completed.append((category, seed))
            else:
                print(f'  FAILED category={category} seed={seed} (exit {result.returncode})')
                failed.append((category, seed))

    print('-' * 60)
    print('SUMMARY')
    print(f'  completed ({len(completed)}): {completed}')
    print(f'  skipped   ({len(skipped)}): {skipped}')
    print(f'  failed    ({len(failed)}): {failed}')

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
