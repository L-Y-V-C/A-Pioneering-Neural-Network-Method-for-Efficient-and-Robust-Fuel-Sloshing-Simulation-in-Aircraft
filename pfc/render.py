#!/usr/bin/env python3
"""Render 3D de rollouts lado a lado (Ground truth | modelo A | modelo B ...) a PNG + GIF (PFC).

Acepta .npz (rollout_dfn.py) y .hdf5 (test de DMCF). El GT se toma del primer archivo.
Ejemplo:
  python pfc/render.py --inputs dfn_sim_0201.npz dmcf/0000/0004.hdf5 --labels DualFluidNet DMCF \
      --out_dir renders/sim_0201 --every 2 --max_frames 100
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from common import load_rollout  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--every', type=int, default=2, help='renderizar 1 de cada N frames')
    p.add_argument('--max_frames', type=int, default=100)
    p.add_argument('--fps', type=int, default=12)
    p.add_argument('--point_size', type=float, default=2.0)
    args = p.parse_args()
    assert len(args.inputs) == len(args.labels)

    rolls = [load_rollout(f) for f in args.inputs]
    gt = rolls[0]['gt']
    seqs = [('Ground truth', gt)] + [(l, r['pred']) for l, r in zip(args.labels, rolls)]
    n = min(args.max_frames + 1, *[len(s) for _, s in seqs])

    # límites fijos según el rango que ocupa el fluido en el GT (+10%), para que no "salte" la cámara
    ref = gt[:n].reshape(-1, 3)
    lo, hi = ref.min(0), ref.max(0)
    margin = 0.1 * (hi - lo).max()
    lo, hi = lo - margin, hi + margin
    zlo, zhi = gt[:n, :, 1].min(), gt[:n, :, 1].max()  # color por altura (eje y = gravedad)

    os.makedirs(args.out_dir, exist_ok=True)
    frames_png = []
    for t in range(0, n, args.every):
        fig = plt.figure(figsize=(4.2 * len(seqs), 4.4), dpi=90)
        for i, (label, seq) in enumerate(seqs):
            ax = fig.add_subplot(1, len(seqs), i + 1, projection='3d')
            pts = seq[t]
            # en los datos de DeepLagrangianFluids la gravedad va en -y: se dibuja y como eje vertical
            ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=args.point_size, c=pts[:, 1],
                       cmap='viridis', vmin=zlo, vmax=zhi, linewidths=0, depthshade=False)
            # partículas fuera del rango del GT (p.ej. si el modelo diverge) se recortan visualmente
            ax.set_clip_on(True)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_zlim(lo[1], hi[1])
            ax.set_box_aspect((hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]))
            ax.view_init(elev=20, azim=-60)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(label, fontsize=12)
        fig.suptitle('frame %d  (t = %.2f s)' % (t, t * 0.02), fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = os.path.join(args.out_dir, 'frame_%04d.png' % t)
        fig.savefig(path)
        plt.close(fig)
        frames_png.append(path)

    imgs = [Image.open(f).convert('P', palette=Image.ADAPTIVE) for f in frames_png]
    gif = os.path.join(args.out_dir, 'rollout.gif')
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / args.fps), loop=0)
    print('frames:', len(frames_png), '->', gif)


if __name__ == '__main__':
    main()
