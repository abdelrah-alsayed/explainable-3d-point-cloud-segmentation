"""
Test a trained part-segmentation model again, with the same test set as its training run.
Based on test_partseg.py by Benny (Nov 2019); rewritten for the runs in log/part_seg.

The run's config.json gives the categories, part filter (full_parts_only / num_parts) and
test cap (test_samples_per_category / test_subset_seed). The test set is built with them, and
its size per category must match the run's sample_counts.csv, or the script stops.

The run's own files are never changed. Results go to <run>/eval/:
    eval_log.txt                 the printed output
    eval_per_category_miou.csv   mIoU per category
    eval_summary.json            all numbers, the settings, and the training's saved numbers

Usage (from the thesis folder):
    python test_partseg.py --log_dir full_parts_per_category/Airplane/seed_42
    python test_partseg.py --log_dir full_parts_joint_model --num_votes 3

With --num_votes 1 (default) the test is the same as the one in train_partseg.py: one pass per
object. The numbers are close to the saved ones but not exactly the same, because the 2,048 points
of each object and the first centre point of the model are picked at random.
"""
import argparse
import csv
import datetime
import importlib
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from data_utils.ShapeNetDataLoader import PartNormalDataset, SEG_CLASSES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

seg_classes = SEG_CLASSES
seg_label_to_cat = {label: cat for cat, labels in seg_classes.items() for label in labels}  # {0: Airplane, ..., 49: Table}
NUM_CLASSES = 16    # length of the category one-hot input of the model
NUM_PART = 50       # part labels over all categories
# Files written by train_partseg.py. This script must never write them.
TRAINING_FILES = {'config.json', 'sample_counts.csv', 'metrics_per_epoch.csv', 'per_category_miou.csv'}


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser('Test a trained PointNet++ part-segmentation run')
    parser.add_argument('--log_dir', type=str, required=True, help='run folder inside --log_root, e.g. full_parts_joint_model')
    parser.add_argument('--log_root', type=str, default=os.path.join(BASE_DIR, 'log', 'part_seg'), help='folder with the runs')
    parser.add_argument('--data_root', type=str, default=os.path.join(BASE_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal'))
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--num_votes', type=int, default=1, help='passes per object, averaged (1 = like training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed, so the same command gives the same numbers')
    parser.add_argument('--out_dir', type=str, default=None, help='where to write the results (default: <run>/eval)')
    return parser.parse_args()


def check_category_numbers(dataset):
    """The model gets the category as a number: the line of the category in synsetoffset2category.txt.
    Every run was trained with the file without Pistol (see data/README.md). Stop if the file differs."""
    in_file = list(dataset.classes_original)
    if set(in_file) != set(seg_classes):
        sys.exit('synsetoffset2category.txt has the categories %s, but the code knows %s. '
                 'Delete the Pistol line (see data/README.md); otherwise Rocket, Skateboard and Table '
                 'get the wrong category number.' % (sorted(in_file), sorted(seg_classes)))


def read_csv(path):
    """csv rows with stripped keys and values (some runs pad them with spaces)"""
    with open(path, newline='') as f:
        return [{k.strip(): (v or '').strip() for k, v in row.items()} for row in csv.DictReader(f)]


def check_test_set(dataset, run_dir, categories):
    """The test set must have as many objects per category as the run's training test set."""
    path = os.path.join(run_dir, 'sample_counts.csv')
    if not os.path.exists(path):
        sys.exit('No sample_counts.csv in %s: cannot check the test set.' % run_dir)
    saved = {row['category']: int(row['test_count']) for row in read_csv(path)}
    now = {cat: 0 for cat in categories}
    for cat, _ in dataset.datapath:
        now[cat] += 1
    if now != saved:
        sys.exit('The test set differs from the training run (sample_counts.csv):\n  now   %s\n  saved %s' % (now, saved))
    return now


def evaluate(classifier, loader, categories, num_votes, device):
    """The test of train_partseg.py, with the option to average several passes per object."""
    total_correct, total_seen = 0, 0
    shape_ious = {cat: [] for cat in categories}
    classifier = classifier.eval()
    with torch.no_grad():
        for points, label, target in tqdm(loader, total=len(loader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(cur_batch_size, NUM_POINT, NUM_PART, device=device)
            for _ in range(num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, NUM_CLASSES))
                vote_pool += seg_pred
            cur_pred_val_logits = (vote_pool / num_votes).cpu().data.numpy()
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            total_correct += np.sum(cur_pred_val == target)
            total_seen += cur_batch_size * NUM_POINT

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = [iou for cat in shape_ious for iou in shape_ious[cat]]
    per_cat = {cat: float(np.mean(v)) for cat, v in shape_ious.items()}
    return {'accuracy': float(total_correct / float(total_seen)),
            'class_avg_iou': float(np.mean(list(per_cat.values()))),
            'instance_avg_iou': float(np.mean(all_shape_ious)),
            'per_category': per_cat}


def main(args):
    run_dir = os.path.join(args.log_root, args.log_dir)
    config_path = os.path.join(run_dir, 'config.json')
    checkpoint_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
    for path in (config_path, checkpoint_path):
        if not os.path.exists(path):
            sys.exit('Not found: %s' % path)
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, 'eval'))
    if out_dir == os.path.abspath(run_dir):
        sys.exit('--out_dir must not be the run folder: the training files there must stay unchanged.')
    os.makedirs(out_dir, exist_ok=True)
    out_files = {'log': 'eval_log.txt', 'csv': 'eval_per_category_miou.csv', 'json': 'eval_summary.json'}
    assert not TRAINING_FILES & set(out_files.values())

    logger = logging.getLogger('Model')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(out_dir, out_files['log']), mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)

    def log_string(text):
        logger.info(text)
        print(text)

    # the settings of the training run; runs made before an option existed did not use it
    with open(config_path) as f:
        config = json.load(f)
    run_args = config.get('args', {})
    setting = lambda key, default=None: config.get(key, run_args.get(key, default))
    categories = setting('categories') or list(seg_classes)
    npoint = setting('npoint', 2048)
    model_name = run_args.get('model', 'pointnet2_part_seg_msg')
    log_string('Run: %s' % run_dir)
    log_string('Categories: %s' % ', '.join(sorted(categories)))
    log_string('full_parts_only=%s  num_parts=%s  test_samples_per_category=%s  test_subset_seed=%s  npoint=%s'
               % (setting('full_parts_only', False), setting('num_parts'), setting('test_samples_per_category'),
                  setting('test_subset_seed', 0), npoint))

    set_seed(args.seed)
    test_dataset = PartNormalDataset(root=args.data_root, npoints=npoint, split='test', normal_channel=False,
                                     class_choice=categories,
                                     samples_per_category=setting('test_samples_per_category'),
                                     subset_seed=setting('test_subset_seed', 0),
                                     full_parts_only=setting('full_parts_only', False),
                                     num_parts=setting('num_parts'), verbose=False)
    check_category_numbers(test_dataset)
    counts = check_test_set(test_dataset, run_dir, categories)
    log_string('Test objects: %d (the same per category as in training)' % len(test_dataset))
    # num_workers=0: the random point picks come from the seeded main process, so the numbers repeat
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = importlib.import_module(model_name)
    classifier = model.get_model(NUM_PART, normal_channel=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    saved_epoch = checkpoint.get('epoch')
    saved_epoch = None if saved_epoch is None else int(saved_epoch) + 1     # the checkpoint counts epochs from 0
    log_string('Model: %s, saved at epoch %s, device %s, votes %d, seed %d'
               % (model_name, saved_epoch, device, args.num_votes, args.seed))

    result = evaluate(classifier, loader, categories, args.num_votes, device)

    # compare with the numbers the training saved for this checkpoint
    saved = {'accuracy': checkpoint.get('test_acc'), 'class_avg_iou': checkpoint.get('class_avg_iou'),
             'instance_avg_iou': checkpoint.get('inctance_avg_iou')}
    saved = {k: (None if v is None else float(v)) for k, v in saved.items()}
    cat_path = os.path.join(run_dir, 'per_category_miou.csv')
    saved_cat = {r['category']: float(r['miou']) for r in read_csv(cat_path)} if os.path.exists(cat_path) else {}
    row = lambda name, now, old: '%-22s %8.4f %8s %9s' % (name, now, '-' if old is None else '%.4f' % old, '-' if old is None else '%+.4f' % (now - old))
    log_string('%-22s %8s %8s %9s' % ('', 'now', 'saved', 'now-saved'))
    for key in ('accuracy', 'class_avg_iou', 'instance_avg_iou'):
        log_string(row(key, result[key], saved[key]))
    for cat in sorted(result['per_category']):
        log_string(row('mIoU ' + cat, result['per_category'][cat], saved_cat.get(cat)))

    with open(os.path.join(out_dir, out_files['csv']), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'miou', 'num_test_samples'])
        for cat in sorted(result['per_category']):
            writer.writerow([cat, result['per_category'][cat], counts[cat]])
    summary = {'run': args.log_dir, 'date': datetime.datetime.now().isoformat(timespec='seconds'),
               'checkpoint_epoch': saved_epoch, 'num_votes': args.num_votes, 'seed': args.seed,
               'device': str(device), 'test_objects': counts, 'result': result,
               'saved_by_training': {**saved, 'per_category': saved_cat}}
    with open(os.path.join(out_dir, out_files['json']), 'w') as f:
        json.dump(summary, f, indent=2)
    log_string('Results written to %s' % out_dir)


if __name__ == '__main__':
    main(parse_args())
