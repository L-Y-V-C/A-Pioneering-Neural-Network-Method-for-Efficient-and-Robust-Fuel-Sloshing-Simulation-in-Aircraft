#!/usr/bin/env python3
"""Resumen de resultados (PFC): métricas comunes de todos los rollouts + JSON de evaluación,
tabla Markdown/CSV y gráfica de error por frame.

Ejemplo:
  python pfc/summarize.py --rollout "DualFluidNet (3.5k it)" sim_0201 results/rollouts/dfn_sim_0201.npz \
      --rollout "DMCF (1.3k it)" sim_0201 .../visual/0000/0004.hdf5 --extra results/dfn_eval.json --out_dir results
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from common import load_rollout, rollout_metrics  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rollout', nargs=3, action='append', metavar=('MODELO', 'ESCENA', 'ARCHIVO'), required=True)
    p.add_argument('--timing', nargs=2, action='append', metavar=('MODELO', 'MS_POR_FRAME'), default=[],
                   help='tiempo de inferencia medido aparte (p.ej. DMCF desde el log de rollout)')
    p.add_argument('--extra', nargs='*', default=[], help='JSON adicionales a copiar al resumen')
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, curves = [], defaultdict(dict)
    for model, scene, path in args.rollout:
        r = load_rollout(path)
        m = rollout_metrics(r['pred'], r['gt'])
        row = {'modelo': model, 'escena': scene, 'frames': m['frames'], 'err_t1_mm': m['err_t1_mm'],
               'err_t2_mm': m['err_t2_mm'], 'd_n_mm': m['d_n_mm'], 'emd_mean': m['emd_mean']}
        if 'times' in r and len(r['times']) > 1:
            row['inferencia_ms'] = float(np.mean(r['times'][1:]) * 1000)
        rows.append(row)
        curves[scene][model] = m['err_mm_per_frame']

    timing = {m: float(t) for m, t in args.timing}
    for row in rows:
        if 'inferencia_ms' not in row and row['modelo'] in timing:
            row['inferencia_ms'] = timing[row['modelo']]

    # promedio por modelo sobre escenas
    by_model = defaultdict(list)
    for row in rows:
        by_model[row['modelo']].append(row)
    avg = []
    for model, rs in by_model.items():
        a = {'modelo': model, 'escena': 'promedio (%d)' % len(rs), 'frames': rs[0]['frames']}
        for k in ('err_t1_mm', 'err_t2_mm', 'd_n_mm', 'emd_mean', 'inferencia_ms'):
            vals = [r[k] for r in rs if r.get(k) is not None]
            a[k] = float(np.mean(vals)) if vals else None
        avg.append(a)

    cols = ['modelo', 'escena', 'frames', 'err_t1_mm', 'err_t2_mm', 'd_n_mm', 'emd_mean', 'inferencia_ms']
    with open(os.path.join(args.out_dir, 'resumen_rollouts.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows + avg:
            w.writerow({c: r.get(c) for c in cols})

    def fmt(v):
        return '-' if v is None else ('%.3f' % v if isinstance(v, float) else str(v))
    md = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    md += ['| ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |' for r in avg + rows]
    extra = {}
    for e in args.extra:
        with open(e) as f:
            extra[os.path.basename(e)] = json.load(f)
    with open(os.path.join(args.out_dir, 'resumen.md'), 'w') as f:
        f.write('# Resumen de rollouts\n\n')
        f.write('err_t1/err_t2/d^n: distancia media por partícula al GT (mm); d^n promedia los n frames del rollout.\n')
        f.write('emd_mean: Wasserstein 1D sobre coordenadas aplanadas (definición de evaluate_network.py de DualFluidNet).\n\n')
        f.write('\n'.join(md) + '\n')
        for k, v in extra.items():
            f.write('\n## %s\n\n```json\n%s\n```\n' % (k, json.dumps(v, indent=1)))
    with open(os.path.join(args.out_dir, 'resumen.json'), 'w') as f:
        json.dump({'rollouts': rows, 'promedios': avg, 'extra': extra}, f, indent=1)

    for scene, ms in curves.items():
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=110)
        for model, err in ms.items():
            ax.plot(np.arange(1, len(err) + 1), err, label=model, lw=1.6)
        ax.set_yscale('log'); ax.set_xlabel('frame del rollout'); ax.set_ylabel('error medio por partícula (mm)')
        ax.set_title('Error acumulado en rollout — %s' % scene); ax.grid(alpha=0.3, which='both'); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, 'error_por_frame_%s.png' % scene)); plt.close(fig)

    print('\n'.join(md))


if __name__ == '__main__':
    main()
