#!/usr/bin/env python3
"""Evaluación n-step de un checkpoint de DualFluidNet sobre el conjunto de validación (PFC).
Usa evaluate_torch() del repo (la misma que corre durante el entrenamiento): err_n1, err_n2 (m) y EMD.

Ejemplo:
  python pfc/eval_dfn.py --ckpt .../ckpt-3500.pt --data_dir /workspace/datasets/ours_6k_box_data --out dfn_eval.json
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from datasets.dataset_reader_physics import read_data_val  # noqa: E402
from evaluate_network import evaluate_torch  # noqa: E402
from models.default_torch import MyParticleNetwork  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', required=True, help='carpeta con valid/')
    p.add_argument('--frame_skip', type=int, default=20, help='igual que en train_network_torch.py')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda')
    model = MyParticleNetwork().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.data_dir, 'valid', '*.zst')))
    ds = read_data_val(files=files, window=1, cache_data=True)
    t0 = time.time()
    with torch.no_grad():
        res = evaluate_torch(model, ds, frame_skip=args.frame_skip, device=device)
    res = {k: (None if v != v else float(v)) for k, v in res.items()}  # nan -> None
    res.update({'err_n1_mm': res['err_n1'] * 1000, 'err_n2_mm': res['err_n2'] * 1000,
                'ckpt': args.ckpt, 'eval_seconds': time.time() - t0, 'valid_files': len(files)})
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
