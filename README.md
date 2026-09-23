# BOLDSimNet

Python implementation of the BOLDSimNet graph-comparison score for
precomputed nonnegative, weighted directed adjacency matrices.

## Installation

```bash
git clone https://github.com/labhai/BOLDSimNet.git
cd BOLDSimNet
python -m pip install -e .
```

Python 3.10 or newer and NumPy are required.

## Usage

```bash
boldsimnet compare A.npy B.npy --labels labels.txt --output result.json
```

`A.npy` and `B.npy` must have the same square shape, finite nonnegative
weights, and a zero diagonal. Matrices use `row=source, column=target`.
`labels.txt` contains one non-empty functional-network label per atlas node.

The JSON result contains `score`, `node_cost`, `centrality_distance`, and
`ordered`. The matrices are compared in the supplied first/second order.
An error is raised when a nonempty graph has no unique principal right
eigenvector direction.

The package implements graph comparison only. It does not perform fMRI
preprocessing, connectivity inference, or group-level analysis.

Python users can call the same implementation directly:

```python
from pathlib import Path
import numpy as np
from boldsimnet import compare

labels = Path("labels.txt").read_text(encoding="utf-8").splitlines()
result = compare(np.load("A.npy"), np.load("B.npy"), labels)
print(result.score)
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## License and citation

The software is distributed under the [MIT License](LICENSE), copyright 2026
Boseong Kim. Citation metadata and the archived preprint reference are provided
in [CITATION.cff](CITATION.cff).
