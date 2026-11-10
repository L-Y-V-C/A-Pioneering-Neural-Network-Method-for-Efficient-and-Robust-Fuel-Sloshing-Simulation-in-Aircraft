#!/usr/bin/env python3
"""Rollout de DualFluidNet sobre una escena de validación (PFC).
Guarda pred/gt/box/tiempos en .npz e imprime métricas + tiempo de inferencia por frame.

Ejemplo:
  python pfc/rollout_dfn.py --ckpt /workspace/runs/dfn_final/train_network_torch_runpod_6k/checkpoints/ckpt-3500.pt \
      --data_dir /workspace/datasets/ours_6k_box_data/valid --scene sim_0201 --frames 100 --out dfn_sim_0201.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))
from common import load_scene, rollout_metrics  # noqa: E402
from models.default_torch import MyParticleNetwork  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='ckpt-*.pt (dict con "model") o *_model_weights*.pt')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--scene', default='sim_0201')
    p.add_argument('--frames', type=int, default=100, help='número de pasos a simular')
    p.add_argument('--out', required=True)
    p.add_argument('--one_step', action='store_true',
                   help='predicción a 1 paso ("teacher forcing"): en cada frame se usa el estado GT como entrada; '
                        'NO es una simulación, sirve para visualizar el error local del modelo')
    args = p.parse_args()

    device = torch.device('cuda')
    scene = load_scene(args.data_dir, args.scene, max_frames=args.frames + 1)
    model = MyParticleNetwork().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()

    box = torch.from_numpy(scene['box']).to(device)
    box_normals = torch.from_numpy(scene['box_normals']).to(device)
    pos = torch.from_numpy(scene['pos'][0]).to(device)
    vel = torch.from_numpy(scene['vel'][0]).to(device)

    preds = [scene['pos'][0]]
    times = []
    with torch.no_grad():
        for t in range(len(scene['pos']) - 1):
            if args.one_step:
                pos = torch.from_numpy(scene['pos'][t]).to(device)
                vel = torch.from_numpy(scene['vel'][t]).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pos, vel = model((pos, vel, None, box, box_normals))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            preds.append(pos.cpu().numpy())

    pred = np.stack(preds)
    times = np.array(times)
    np.savez_compressed(args.out, pred=pred, gt=scene['pos'], box=scene['box'], times=times)

    m = rollout_metrics(pred, scene['pos'])
    m.update({
        'model': 'DualFluidNet', 'ckpt': args.ckpt, 'scene': args.scene,
        'mode': 'one_step (teacher forcing)' if args.one_step else 'rollout',
        'particles': int(pred.shape[1]),
        # se descarta el primer paso (calentamiento de CUDA)
        'inference_ms': float(times[1:].mean() * 1000) if len(times) > 1 else float(times.mean() * 1000),
    })
    with open(os.path.splitext(args.out)[0] + '_metrics.json', 'w') as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: v for k, v in m.items() if not k.endswith('per_frame')}, indent=1))


if __name__ == '__main__':
    main()
