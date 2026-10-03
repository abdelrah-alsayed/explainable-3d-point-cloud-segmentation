# *_*coding:utf-8 *_*
import json
import os
import warnings

import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


# Mapping from category ('Chair') to its list of global segmentation part IDs.
# The length is the category's *nominal* part count (used by full_parts_only).
SEG_CLASSES = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    def __init__(
        self,
        root="./data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        npoints=2500,
        split="train",
        class_choice=None,
        normal_channel=False,
        samples_per_category=None,
        subset_seed=0,
        full_parts_only=False,
        num_parts=None,
        verbose=True,
    ):
        self.npoints = npoints
        self.root = root
        self.verbose = verbose
        self.catfile = os.path.join(self.root, "synsetoffset2category.txt")
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, "r") as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(
            os.path.join(
                self.root, "train_test_split", "shuffled_train_file_list.json"
            ),
            "r",
        ) as f:
            train_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        with open(
            os.path.join(self.root, "train_test_split", "shuffled_val_file_list.json"),
            "r",
        ) as f:
            val_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        with open(
            os.path.join(self.root, "train_test_split", "shuffled_test_file_list.json"),
            "r",
        ) as f:
            test_ids = set([str(d.split("/")[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == "trainval":
                fns = [
                    fn
                    for fn in fns
                    if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))
                ]
            elif split == "train":
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == "val":
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == "test":
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print("Unknown split: %s. Exiting.." % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]
                self.meta[item].append(os.path.join(dir_point, token + ".txt"))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        # Optionally keep only objects with a target number of distinct parts.
        # num_parts=k  -> exactly k parts (constant across categories, e.g. only
        #                 3-part objects regardless of the category's nominal max).
        # full_parts_only -> the category's full nominal set (k varies per category).
        # Done BEFORE capping so the cap selects only among the kept objects.
        if num_parts is not None or full_parts_only:
            self.datapath = self._filter_by_parts(self.datapath, num_parts)

        if samples_per_category is not None:
            grouped = {}
            for cat, fn in self.datapath:
                grouped.setdefault(cat, []).append(fn)

            if samples_per_category == "min":
                min_cat = min(grouped, key=lambda c: len(grouped[c]))
                cap = len(grouped[min_cat])
                print(f"[PartNormalDataset] samples_per_category='min': cap={cap} (set by '{min_cat}')")
            else:
                cap = int(samples_per_category)

            rng_subset = np.random.RandomState(subset_seed)
            new_datapath = []
            for cat in sorted(grouped.keys()):
                entries = grouped[cat]
                if len(entries) <= cap:
                    if len(entries) < cap:
                        warnings.warn(
                            f"[PartNormalDataset] '{cat}' has {len(entries)} samples < cap {cap}; keeping all."
                        )
                    chosen = entries
                else:
                    idx = sorted(rng_subset.choice(len(entries), cap, replace=False))
                    chosen = [entries[i] for i in idx]
                new_datapath.extend((cat, fn) for fn in chosen)
                print(f"  {cat}: {len(chosen)}")
            self.datapath = new_datapath

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = SEG_CLASSES

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    # --- distinct-part counting (shared, cached on disk) ---------------------
    # Counting distinct part labels means scanning each .txt file, so results
    # are cached on disk (keyed by file path) to keep repeated runs cheap.

    def _part_cache_path(self):
        return os.path.join(self.root, ".part_count_cache.json")

    def _load_part_cache(self):
        try:
            with open(self._part_cache_path(), "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_part_cache(self, cache):
        try:
            with open(self._part_cache_path(), "w") as f:
                json.dump(cache, f)
        except Exception:
            pass

    def _distinct_part_count(self, fn, cache):
        n = cache.get(fn)
        if n is None:
            labels = set()
            with open(fn, "r") as f:
                for line in f:
                    if line.strip():
                        labels.add(line.rsplit(None, 1)[-1])  # last col = part label
            n = len(labels)
            cache[fn] = n
        return n

    def _filter_by_parts(self, datapath, num_parts):
        """Keep objects whose distinct-part count matches a target.

        num_parts=k -> keep objects with exactly k distinct parts.
        num_parts=None -> keep objects with the category's full nominal count.
        """
        cache = self._load_part_cache()
        n_before = len(cache)
        kept, dropped = {}, {}
        new_datapath = []
        for cat, fn in datapath:
            n_parts = self._distinct_part_count(fn, cache)
            target = num_parts if num_parts is not None else len(SEG_CLASSES[cat])
            if n_parts == target:
                new_datapath.append((cat, fn))
                kept[cat] = kept.get(cat, 0) + 1
            else:
                dropped[cat] = dropped.get(cat, 0) + 1

        if len(cache) != n_before:
            self._save_part_cache(cache)

        if self.verbose:
            what = f"exactly {num_parts}" if num_parts is not None else "full nominal"
            for cat in sorted(set(kept) | set(dropped)):
                target = num_parts if num_parts is not None else len(SEG_CLASSES[cat])
                print(
                    f"[PartNormalDataset] parts={what} '{cat}': "
                    f"kept {kept.get(cat, 0)} (={target} parts), "
                    f"dropped {dropped.get(cat, 0)}"
                )
        return new_datapath

    def part_count_histogram(self):
        """Return {category: {k: n_objects_with_k_distinct_parts}} over this
        split's datapath. Utility for dataset exploration / cap sizing."""
        from collections import Counter, defaultdict

        cache = self._load_part_cache()
        n_before = len(cache)
        hist = defaultdict(Counter)
        for cat, fn in self.datapath:
            hist[cat][self._distinct_part_count(fn, cache)] += 1
        if len(cache) != n_before:
            self._save_part_cache(cache)
        return hist

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)
