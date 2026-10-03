"""One explicit study configuration; final sampling budget is chosen after trial."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class StudyConfig:
    category: str = "Airplane"
    log_dir: str = "log/part_seg/parts4_cap596/Airplane/seed_31337"
    output_dir: str = "log/part_seg/parts4_cap596/Airplane/seed_31337/explain/study_v2"
    npoints: int = 2048
    num_parts_filter: Optional[int] = 4
    test_count: int = 118
    subset_seed: int = 0
    sample_seed: int = 0
    region_seed: int = 0
    groups: int = 32
    trial_objects: list = field(default_factory=lambda: [0,13,26,39,52,65,78,91,104,117])
    example_objects: list = field(default_factory=lambda: [0,65,117])
    exact_objects: list = field(default_factory=lambda: [0,65,117])
    exact_groups: int = 8
    exact_budgets: list = field(default_factory=lambda: [64,128,254])
    trial_budgets: list = field(default_factory=lambda: [2048,4096,8192])
    kernel_seeds: list = field(default_factory=lambda: [0,1,2])
    replacement_seeds: list = field(default_factory=lambda: [0,1,2])
    batch_size: int = 8
    score_chunk_size: int = 64
    random_removal_orders: int = 5
    device: str = "auto"
    # Legacy studies retain their original score. Final-study configs opt in explicitly.
    logprob_mode: str = "raw50"
    object_names: list = field(default_factory=list)
    training_seen_objects: list = field(default_factory=list)
    selection_seed: int = 0
    cross_mask_removal: bool = False

    @classmethod
    def load(cls, path=None):
        cfg = cls(**json.loads(Path(path).read_text())) if path else cls()
        cfg.validate()
        return cfg

    def validate(self):
        if not isinstance(self.selection_seed,int) or self.selection_seed < 0:
            raise ValueError("Selection seed must be a nonnegative integer")
        if self.logprob_mode not in ("raw50", "category"):
            raise ValueError("Unknown log-probability definition")
        if self.object_names and (len(self.object_names) != self.test_count or
                                  len(set(self.object_names)) != self.test_count or
                                  any(Path(n).name != n or not n.endswith('.txt') for n in self.object_names)):
            raise ValueError("Object names must be unique source filenames, one per object")
        if (len(set(self.training_seen_objects)) != len(self.training_seen_objects) or
                not set(self.training_seen_objects) <= set(self.object_names)):
            raise ValueError("Training-seen exceptions must name explicit selected objects")
        if not 2 <= self.groups <= min(self.npoints, 64):
            raise ValueError("groups must be between 2 and min(npoints,64)")
        if not 2 <= self.exact_groups <= 12:
            raise ValueError("Exact reference must have 2–12 groups")
        if self.test_count < 1 or min(self.batch_size, self.score_chunk_size, self.random_removal_orders) < 1:
            raise ValueError("Counts and batch sizes must be positive")
        for name in ("trial_objects", "example_objects", "exact_objects"):
            ids = getattr(self, name)
            if not ids or len(ids) != len(set(ids)) or any(i < 0 or i >= self.test_count for i in ids):
                raise ValueError(f"Invalid {name}")
        if not set(self.exact_objects+self.example_objects) <= set(self.trial_objects):
            raise ValueError("Exact and example objects must be selected from trial objects")
        if any(self.is_training_seen(i) for i in self.trial_objects):
            raise ValueError("Extra checks must use only main-study objects")
        for budgets, groups in ((self.trial_budgets, self.groups), (self.exact_budgets, self.exact_groups)):
            if not budgets or sorted(set(budgets)) != budgets or min(budgets) < groups+1:
                raise ValueError("Budgets must be increasing, unique, and at least groups+1")
        for seeds in (self.kernel_seeds, self.replacement_seeds):
            if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**32-1 for s in seeds):
                raise ValueError("Seeds must be distinct nonnegative 32-bit integers")

    def resolve(self, path):
        p = Path(path).expanduser()
        return p if p.is_absolute() else ROOT/p

    def is_training_seen(self, index):
        return bool(self.object_names and self.object_names[index] in self.training_seen_objects)

    def cohort(self, index):
        return "training_seen" if self.is_training_seen(index) else "main"

    def as_dict(self):
        return asdict(self)
