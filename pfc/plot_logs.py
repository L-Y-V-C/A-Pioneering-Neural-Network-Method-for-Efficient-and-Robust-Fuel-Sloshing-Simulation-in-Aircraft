#!/usr/bin/env python3
"""Exporta curvas de entrenamiento (TensorBoard) a PNG + CSV (PFC).
Lee tanto escalares clásicos (DualFluidNet, torch SummaryWriter) como tensores tf.summary (DMCF).

Ejemplo:
  python pfc/plot_logs.py --logdir /workspace/runs/dfn_final --name DualFluidNet --out_dir results/curves
  python pfc/plot_logs.py --logdir /workspace/runs/dmcf_final/train_log --name DMCF --out_dir results/curves
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402
from tensorboard.util import tensor_util  # noqa: E402


def read_events(logdir):
    data = defaultdict(list)
    files = glob.glob(os.path.join(logdir, '**', 'events.out.tfevents.*'), recursive=True)
    for f in sorted(files):
        acc = EventAccumulator(f, size_guidance={'scalars': 0, 'tensors': 0})
        acc.Reload()
        for tag in acc.Tags().get('scalars', []):
            data[tag] += [(e.step, e.value) for e in acc.Scalars(tag)]
        for tag in acc.Tags().get('tensors', []):
            for e in acc.Tensors(tag):
                try:
                    v = float(np.asarray(tensor_util.make_ndarray(e.tensor_proto)).ravel()[0])
                except Exception:
                    continue
                data[tag].append((e.step, v))
    return {k: sorted(set(v)) for k, v in data.items() if v}


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return np.asarray(y)
    return np.convolve(y, np.ones(w) / w, mode='valid')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--smooth', type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = read_events(args.logdir)
    skip = ('runstats/', 'DataLatency', 'train/timesteps', 'train/warmup')
    data = {k: v for k, v in data.items() if not k.startswith(skip)}
    print('tags:', {k: len(v) for k, v in data.items()})
    with open(os.path.join(args.out_dir, '%s_scalars.csv' % args.name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tag', 'step', 'value'])
        for tag, vals in data.items():
            for s, v in vals:
                w.writerow([tag, s, v])

    for tag, vals in data.items():
        if len(vals) < 2:
            continue
        steps, ys = zip(*vals)
        fig, ax = plt.subplots(figsize=(6, 3.6), dpi=110)
        ax.plot(steps, ys, alpha=0.3 if len(ys) > args.smooth * 2 else 1.0, lw=0.8)
        if len(ys) > args.smooth * 2:
            ys_s = smooth(ys, args.smooth)
            ax.plot(steps[args.smooth - 1:], ys_s, lw=1.6, label='media móvil (%d)' % args.smooth)
            ax.legend()
        ax.set_xlabel('step'); ax.set_ylabel(tag); ax.set_title('%s — %s' % (args.name, tag))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, '%s_%s.png' % (args.name, tag.replace('/', '_'))))
        plt.close(fig)
    print('-> %s' % args.out_dir)


if __name__ == '__main__':
    main()
