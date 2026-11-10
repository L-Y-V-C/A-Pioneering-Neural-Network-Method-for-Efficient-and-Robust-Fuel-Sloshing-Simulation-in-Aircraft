"""Utilidades compartidas (PFC): lectura de escenas DeepLagrangianFluids, carga de
rollouts (npz de DualFluidNet / hdf5 de DMCF) y métricas comunes a ambos modelos."""
import glob
import os

import msgpack
import msgpack_numpy
import numpy as np
import zstandard as zstd
from scipy.stats import wasserstein_distance

msgpack_numpy.patch()


def load_scene(data_dir, scene, max_frames=None):
    """Concatena los archivos <scene>_XX.msgpack.zst (en orden) de una escena.
    Devuelve dict con pos (T,P,3), vel (T,P,3), box (B,3), box_normals (B,3)."""
    files = sorted(glob.glob(os.path.join(data_dir, scene + '_*.msgpack.zst')))
    assert files, 'no hay archivos para %s en %s' % (scene, data_dir)
    dec = zstd.ZstdDecompressor()
    frames = []
    for f in files:
        with open(f, 'rb') as fh:
            frames += msgpack.unpackb(dec.decompress(fh.read()), raw=False)
        if max_frames is not None and len(frames) >= max_frames:
            break
    frames = sorted(frames, key=lambda d: d['frame_id'])
    if max_frames is not None:
        frames = frames[:max_frames]
    assert frames[0]['frame_id'] == 0, 'la escena no empieza en el frame 0'
    first = next(d for d in frames if 'box' in d)
    return {
        'pos': np.stack([d['pos'] for d in frames]).astype(np.float32),
        'vel': np.stack([d['vel'] for d in frames]).astype(np.float32),
        'box': first['box'].astype(np.float32),
        'box_normals': first['box_normals'].astype(np.float32),
    }


def load_rollout(path):
    """Carga un rollout como dict {pred, gt, box, [times]} desde .npz (DFN) o .hdf5 (DMCF)."""
    if path.endswith('.npz'):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    import h5py
    with h5py.File(path, 'r') as f:
        grp = f[list(f.keys())[0]]
        return {'pred': grp['pred'][()], 'gt': grp['gt'][()], 'box': grp['bnd'][()]}


def rollout_metrics(pred, gt):
    """Métricas comunes sobre un rollout (frame 0 = condición inicial, se excluye).
    - err_mm[t]: distancia euclidiana media por partícula al GT, en mm.
    - d_n: promedio de err_mm sobre los n frames (d^n del paper de DualFluidNet).
    - emd[t]: Wasserstein 1D sobre coordenadas aplanadas (misma definición que evaluate_network.py de DFN)."""
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    err = np.linalg.norm(pred[1:] - gt[1:], axis=-1).mean(axis=-1) * 1000.0
    emd = np.array([wasserstein_distance(pred[t].ravel(), gt[t].ravel()) for t in range(1, n)])
    return {
        'frames': int(n - 1),
        'err_t1_mm': float(err[0]),
        'err_t2_mm': float(err[1]) if n > 2 else None,
        'd_n_mm': float(err.mean()),
        'emd_mean': float(emd.mean()),
        'err_mm_per_frame': err.tolist(),
        'emd_per_frame': emd.tolist(),
    }
