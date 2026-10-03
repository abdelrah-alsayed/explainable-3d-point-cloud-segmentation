# Data

The ShapeNetPart dataset is not included in this repository because of its large size.

## Download

Download the dataset from:

https://omnomnom.vision.rwth-aachen.de/data/point2vec/data/

## Setup

Extract the archive into the `data/` directory.

Expected structure:

```text
data/
└── shapenetcore_partanno_segmentation_benchmark_v0_normal/
    ├── 02691156/   # Airplane
    ├── 02773838/   # Bag
    ├── ...         # 16 category folders
    └── synsetoffset2category.txt
```

Make sure all 16 category folders are present after extraction.

Note: this project uses 15 of the 16 categories. Pistol is excluded for ethical reasons (Brandenburg Higher Education Act, BbgHG §§ 3(3) and 70(3)): after extraction, delete the line `Pistol	03948459` from `synsetoffset2category.txt`, otherwise Rocket, Skateboard and Table get the wrong category number.

See the main README for complete setup and training instructions.
