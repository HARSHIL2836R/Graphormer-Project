# Setup

The repository ships **without** the pretrained weights (~193 MB) or the
PCQM4Mv2 raw dataset (~60 MB compressed). This file walks through getting
both onto your machine.

---

## 1. Install Python dependencies

```bash
pip install torch numpy ogb torch_geometric rdkit
```

Tested on Python 3.10–3.12, `torch` 2.1+, `ogb` 1.3.6.

---

## 2. Get the pretrained checkpoint

Download link:

```
https://zenodo.org/records/8399738/files/checkpoint_best_pcqm4mv2.pt?download=1
```

Save the file as `checkpoint_best_pcqm4mv2.pt` inside the `mini_graphormer/`
folder, then pass the local path to any script as,

```bash
python validate.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 1000
```

---

## 3. Get the PCQM4Mv2 raw dataset

The streaming loader in `validate.py` reads two files directly from disk:

```
dataset/pcqm4m-v2/raw/data.csv.gz   (~55 MB compressed CSV of all 3.7M rows)
dataset/pcqm4m-v2/split_dict.pt     (~33 MB train/valid/test indices)
```

Run this once from inside `mini_graphormer/`:

```bash
python -c "
import ogb.utils
from ogb.utils.mol import smiles2graph
ogb.utils.smiles2graph = smiles2graph     # patch ogb 1.3.6 internal import bug

from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
try:
    PygPCQM4Mv2Dataset(root='dataset')
except Exception as e:
    print(f'OGB processing crashed (expected): {type(e).__name__}: {e}')

import shutil
shutil.rmtree('dataset/pcqm4m-v2/processed', ignore_errors=True)
print('Done.')
"
```

```bash
ls -la dataset/pcqm4m-v2/raw/data.csv.gz dataset/pcqm4m-v2/split_dict.pt
```

You should see roughly `54 MB` and `33 MB`.

---

## 4. Verify everything works

```bash
python validate.py --ckpt checkpoint_best_pcqm4mv2.pt --skip-mae
```

Expected output:

```
TEST 1 — strict checkpoint load
  ...
  state_dict has 210 tensors
  weight stats (after load):
    atom_encoder             shape=(4609, 768)          mean=+2.3791e-05  std=2.2329e-02
    spatial_pos_encoder      shape=(512, 32)            mean=-8.6182e-03  std=1.4132e-01
    layer0.q_proj            shape=(768, 768)           mean=-9.1567e-06  std=4.8459e-02
    embed_out                shape=(1, 768)             mean=-3.4584e-03  std=2.2480e-02
  PASS — strict load succeeded with 0 missing / 0 unexpected keys
```

