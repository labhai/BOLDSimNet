# BOLDSimNet

This repository provides a Python implementation of BOLDSimNet.


## Installation

```bash
git clone https://github.com/labhai/BOLDSimNet.git
cd BOLDSimNet
python -m pip install -e .
```

Python 3.10 or newer and NumPy are required.

## Python API

```python
import numpy as np

from boldsimnet import compare

first = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
        [3.0, 0.0, 0.0],
    ]
)
second = np.array(
    [
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 2.0],
        [2.5, 0.0, 0.0],
    ]
)
labels = ("Visual", "Somatomotor", "Control")

result = compare(first, second, labels)
print(result.score)
print(result.node_cost)            # NC in the manuscript
print(result.centrality_distance)  # ED in the manuscript
```

Adjacency matrices use `row=source, column=target`. They must have the same
square shape, finite nonnegative weights, and an exactly zero diagonal. Supply
one non-empty functional-network label per atlas node.

## Command line

The CLI accepts two `.npy` matrices and a UTF-8 label file containing one label
per line:

```bash
boldsimnet compare A.npy B.npy \
  --labels resources/schaefer100_yeo7_labels.txt \
  --output result.json
```

The supplied label file follows the Schaefer-100 parcel order and Yeo-7
network assignment used in the manuscript. Other atlases require labels in
their own adjacency-matrix order.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License and citation

The software is distributed under the [MIT License](LICENSE), copyright 2026
Boseong Kim. Atlas-label attribution is provided in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Source repository: https://github.com/labhai/BOLDSimNet

Please cite the archived preprint:

> Kim, B., Chakladar, D. D., Chung, H., & Jang, I. (2025). BOLDSimNet:
> Examining brain network similarity between task and resting-state fMRI.
> *arXiv*. https://doi.org/10.48550/arXiv.2504.01274

```bibtex
@article{kim2025boldsimnet,
  title   = {BOLDSimNet: Examining Brain Network Similarity between Task and
             Resting-State fMRI},
  author  = {Kim, Boseong and Chakladar, Debashis Das and Chung, Haejun and
             Jang, Ikbeom},
  journal = {arXiv preprint arXiv:2504.01274},
  year    = {2025},
  doi     = {10.48550/arXiv.2504.01274},
  url     = {https://arxiv.org/abs/2504.01274}
}
```

Machine-readable citation metadata are provided in [CITATION.cff](CITATION.cff).
